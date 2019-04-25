from bert_embedding import BertEmbedding
import numpy as np
import mxnet as mx


class BertSimilarity():
    def __init__(self):
        ctx = mx.gpu(0)
        self.bert = BertEmbedding(ctx=ctx)

    def get_similarity(self, sentence1, sentence2):
        embeddings = self.bert([sentence1, sentence2])
        np.avg(embeddings)
        return np.cosine(embeddings[0], embeddings[1]