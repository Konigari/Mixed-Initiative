import collections
import numpy as np
import os
import copy
import pdb
from functools import reduce
class Data():
    def __init__(self, H, Q, A):
        self.h = H
        self.q = Q
        self.a = A
        self.datalen = len(self.h)
        self.batch_startidx = 0

    def next_batch(self, batch_size, shuffle=True):
        finishOneEpoch = False
        def shuffle_data():
            perm = np.arange(self.datalen)
            np.random.shuffle(perm)
            self.h = self.h[perm]
            self.q = self.q[perm]
            self.a = self.a[perm]

        if shuffle and self.batch_startidx == 0:
            shuffle_data()

        if self.batch_startidx + batch_size > self.datalen:
            batch_numrest = self.datalen - self.batch_startidx
            rest_H = self.h[self.batch_startidx:]
            rest_Q = self.q[self.batch_startidx:]
            rest_A = self.a[self.batch_startidx:]
            if shuffle:
                shuffle_data()
            num_next = batch_size - batch_numrest
            next_H = self.h[0:num_next]
            next_Q = self.q[0:num_next]
            next_A = self.a[0:num_next]
            batch_h = np.concatenate([rest_H, next_H], axis = 0)
            batch_q = np.concatenate([rest_Q, next_Q], axis = 0)
            batch_a = np.concatenate([rest_A, next_A], axis = 0)
            self.batch_startidx = num_next
            finishOneEpoch = True
        else:
            batch_h = self.h[self.batch_startidx:self.batch_startidx + batch_size]
            batch_q = self.q[self.batch_startidx:self.batch_startidx + batch_size]
            batch_a = self.a[self.batch_startidx:self.batch_startidx + batch_size]
            self.batch_startidx += batch_size
        return batch_h, batch_q, batch_a, finishOneEpoch


class GenData():
    def __init__(self, datadir, memory_size = 5, description_size = 10, vocab_size = -1, taskid = -1):
        self.fvocab = os.path.join(datadir, 'vocab.out')
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.description_size = description_size
        self.get_data_from_raw(datadir, taskid)

    def get_data_from_raw(self, datadir, taskid = -1):
        train_raw = []
        test_raw = []
        nload = 0
        filenameformat = "qa{}_".format(taskid)
        for file in os.listdir(datadir):
            if taskid == -1 or filenameformat in file and 'train' in file:
                print(file)
                train_raw += self.read_source(os.path.join(datadir, file))
            elif taskid == -1 or filenameformat in file and 'test' in file:
                print (file)
                test_raw += self.read_source(os.path.join(datadir, file))

        self.statistics(train_raw)
        self.vocab = self.buildDict(train_raw)

        H, Q, A = self.vectorize(train_raw)
        self.train = Data(H, Q, A)

        H, Q, A = self.vectorize(test_raw)
        self.test = Data(H, Q, A)

    def read_source(self, file):
        def parse_line(line):
            line = line.strip()
            if line[-1] == '.' or '?':
                line = line[0:-1]
            line = line.split(' ')
            ID = line[0]
            line = line[1:]
            return ID, line

        history = []
        dataset = []
        with open(file) as f:
            for line in f:
                line = line.lower().strip()
                if len(line.split('\t')) == 3: # is_question(line):
                    # print(line)
                    q, a, _ = line.split('\t')
                    # print(q,a)
                    _, q = parse_line(q)
                    _history = copy.deepcopy(history)
                    # print(a)
                    dataset.append((_history, q, [a]))
                else:
                    id, line = parse_line(line)
                    if id == '1':
                        history = []
                    history.append(line)

        return dataset

    def statistics(self, train_raw):
        history_len = [len(h) for h,q,a in train_raw]
        self.max_history_length = max(history_len)
        self.mean_history_length = np.mean(history_len)

        self.max_history_sentence_length = max([len(x) for x in reduce(lambda x,y:x+y, [h for h,q,ans in train_raw])])
        self.max_query_sentence_length = max([len(q) for h,q,a in train_raw])
        print('===============================================')
        print('\t\tmax_history_length: {0}\n\
\t\tmean_history_length: {1}\n\
\t\tmax_history_sentence_length: {2}\n\
\t\tmax_query_sentence_length:{3}'.format(
                self.max_history_length,
                self.mean_history_length,
                self.max_history_sentence_length,
                self.max_query_sentence_length))

    def buildDict(self, train_raw):
        summ = lambda x,y : x+y
        allwords = reduce(summ, [reduce(summ, h) + q + a for h,q,a in train_raw])
        vocab = collections.Counter(allwords)
        vocab_sort = sorted(vocab, key = vocab.get, reverse = True)
        # print vocabulary to file
        with open(self.fvocab, 'w') as f:
            for word in vocab_sort:
                print('\t'.join([word, str(vocab[word])]),file=f)
        print ('===============================================')
        print ('written vocabulary to ' + self.fvocab)
        self.vocab_size = self.vocab_size if self.vocab_size != -1 else len(vocab_sort) + 2
        vocab = zip(vocab_sort[0: self.vocab_size - 2], range(1, self.vocab_size - 1))

        # add <unk> and <nil> to vocabulary
        vocab = list(vocab)
        vocab.append(('<nil>', 0))
        vocab.append(('<unk>', self.vocab_size - 1))
        assert self.vocab_size == len(vocab)
        print ('vocabulary size:', self.vocab_size)
        return dict(vocab)

    def vectorize(self, raw):

        def complete_sentence(sent, length):
            """
            complete a sentence to specific "length" with <nil> and turn it into IDs
            """
            sent = [self.vocab.get(w, self.vocab['<unk>']) for w in sent]
            if len(sent) > length:
                sent = sent[0:length]
            else:
                sent += [self.vocab['<nil>']] * (length - len(sent))
            return sent

        H = []
        Q = []
        A = []
        for rawH, rawQ, rawA in raw:
            #deal with history
            idxH = copy.deepcopy(rawH)
            if len(idxH) > self.memory_size: # only remain the lastest memory_size ones
                idxH = idxH[len(idxH) - self.memory_size : ]
            for idx, h in enumerate(idxH):
                idxH[idx] = complete_sentence(h, self.description_size)
            idxH += [[self.vocab['<nil>']] * self.description_size for _ in range(self.memory_size - len(idxH))]

            #deal with question
            idxQ = complete_sentence(rawQ, self.description_size)

            #deal with answer
            idxA = [0] * len(self.vocab)
            idxA[self.vocab.get(rawA[0], self.vocab['<unk>'])] = 1

            H.append(idxH)
            Q.append(idxQ)
            A.append(idxA)

        return np.asarray(H), np.asarray(Q), np.asarray(A)

if __name__ == '__main__':
    data = GenData('../data/tasks_1-20_v1-2/en', memory_size = 15, description_size = 10, vocab_size = -1, taskid = 6)
    for i in range(10):
        batchh, batchq, batcha, finishOneEpoch = data.train.next_batch(1200, shuffle=False)


