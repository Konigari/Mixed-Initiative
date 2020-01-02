import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import pdb

class MemN2N(object):
    def __init__(self, config, sess):
        self.sess = sess

        self.batch_size = config.batch_size
        self.n_emb = config.emb_size
        self.n_memory = config.memory_size
        self.n_sentence = config.description_length
        self.n_vocab = config.vocab_size

        self.learning_rate = config.learning_rate
        self.num_epoch = config.num_epoch

        self.max_grad_norm = 40

        self.set_inputs_and_variables()
        self.logs_path = "/tmp/memnet"

    def set_inputs_and_variables(self):
        # define inputs
        self.in_history = tf.placeholder(tf.int32, [None, self.n_memory, self.n_sentence], name = "history")
        self.in_query = tf.placeholder(tf.int32, [None, self.n_sentence], name = "query")
        self.in_answer = tf.placeholder(tf.int32, [None, self.n_vocab], name = "answer")

        # define variables
        self.weights = {}

        def concat_nil_random():
            nil_embedding = tf.zeros([1, self.n_emb])
            concat_emb = tf.concat([nil_embedding, tf.random_normal([self.n_vocab - 1, self.n_emb], stddev=0.1)],0 )
            
            return concat_emb

        self.weights["embA"] = tf.Variable(concat_nil_random(), name = "embA")
        self.weights["embB"] = self.weights["embA"]
        self.weights["embC"] = tf.Variable(concat_nil_random(), name = "embC")
        self.weights["W"] = tf.Variable(concat_nil_random(), name = "W")

    def forward(self, history, query):
        m_embA = tf.nn.embedding_lookup(self.weights["embA"], history)  # A * x_{ij}
        u_emb = tf.nn.embedding_lookup(self.weights["embB"], query) # B * q_{ij}
        """
        shape of m_embA and u_emb:
            m_embA: (batch_size, self.n_memory, self. self.n_sentence, self.n_emb)
            u_emb: (batch_size, self.n_sentence, self.n_emb)
        should be:
            mA: (batch_size, self.n_memory, self.n_emb)
            u: (batch_size, self.n_emb)
        """
        mA = tf.reduce_sum(m_embA, 2) # history embedding:  m_i=\sum_j{A x_{ij}}   shape:(batch_size, self.n_memory, self.n_emb)
        u = tf.reduce_sum(u_emb, 1) # query embedding:    u=\sum_j{B q_{ij}}     shape:(batch_size, self.n_emb)
        u_tmp = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])                     #   shape:(batch_size, 1, self.n_emb)

        # get probability: match between utterance and memory
        probs = tf.nn.softmax(tf.reduce_sum(mA * u_tmp, 2))  # shape of m * u_tmp: (batch_size, self.n_memory, self.n_emb), shape of probs: (batch_size, self.n_memory)

        # get memory representation
        m_embC = tf.nn.embedding_lookup(self.weights["embC"], history)  # C * x_{ij}
        mC = tf.reduce_sum(m_embC, 2) # shape:(batch_size, self.n_memory, self.n_emb)
        o = tf.reduce_sum(tf.expand_dims(probs, -1) * mC, 1) # shape:(batch_size, self.n_emb)

        # get prediction before softmax
        a = tf.matmul(o + u, tf.transpose(self.weights["W"], [1, 0]))
        return a

    def defgradient(self):
        # define cost
        self.prediction = self.forward(self.in_history, self.in_query)
        self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                logits = self.prediction,
                labels = tf.cast(self.in_answer, dtype=tf.float32)), name = "cost")

        correct_prediction = tf.equal(tf.argmax(self.in_answer, 1), tf.argmax(self.prediction, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))

        # cannot minimize cost directly, because nil items should be fixed
        self._nil_vars = set([self.weights["embA"].name, self.weights["embC"].name, self.weights['W'].name])

        # define gradient
        def fillnil(grad):
            grad = tf.convert_to_tensor(grad)
            nil_embedding = tf.zeros([1, int(grad.get_shape()[1])])
            grad_rest = tf.slice(grad, [1,0], [-1, -1])
            return tf.concat([nil_embedding, grad_rest],0 )

        self.finalgrads = []
        self.gradvalue = {}
        gradFromCost = tf.train.GradientDescentOptimizer(self.learning_rate).compute_gradients(self.cost)
        #gradFromCost = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in gradFromCost]
        for g, v in gradFromCost:
            if v.name in self._nil_vars:
                print('v.name: ', v.name)
                g_ = fillnil(g)
                self.gradvalue[v.name] = tf.reduce_mean(g_)
                self.finalgrads.append((g_, v))
            else:
                self.finalgrads.append((g, v))
        self.train_opt = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(self.finalgrads)


    def train(self, data):
        # run
        init = tf.initialize_all_variables()
        self.sess.run(init)

        self.defgradient()
        writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("embAgrad", self.gradvalue[self.weights["embA"].name])
        tf.summary.scalar("embCgrad", self.gradvalue[self.weights["embC"].name])
        summary_op = tf.summary.merge_all()

        for epoch in range(self.num_epoch):
            epochCost = 0
            batch_idx = 0
            while True:
                batch_idx += 1
                batchh, batchq, batcha, finishOneEpoch = data.train.next_batch(self.batch_size)
                if finishOneEpoch:
                    break
                dataTofeed = {self.in_history: batchh,
                              self.in_query: batchq,
                              self.in_answer: batcha}

                _, summary = self.sess.run([self.train_opt, summary_op], feed_dict = dataTofeed)
                writer.add_summary(summary, epoch * self.batch_size + batch_idx)

                epochCost += self.sess.run(self.cost, feed_dict = dataTofeed)

            epochTrainAccuracy = self.sess.run(self.accuracy, feed_dict = {self.in_history: data.train.h,
                                                                      self.in_query: data.train.q,
                                                                      self.in_answer: data.train.a})
            epochTestAccuracy = self.sess.run(self.accuracy, feed_dict = {self.in_history: data.test.h,
                                                                      self.in_query: data.test.q,
                                                                      self.in_answer: data.test.a})
            print('Epoch {0}: \nTraining Accuracy: {1} Test Accuracy: {2} Cost: {3}'.format(epoch,
                                        round(epochTrainAccuracy, 4),
                                        round(epochTestAccuracy, 4),
                                        round(epochCost, 4)))
            