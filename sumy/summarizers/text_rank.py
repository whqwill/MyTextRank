# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import numpy
except ImportError:
    numpy = None

from ._summarizer import AbstractSummarizer

def randomEmbedding(size):
    return np.array([(np.random.random()-0.5)/size for i in range(size)])

def count_tf(words):
    d = {}
    for word in words:
        if not word in d:
            d[word] = 1
        else:
            d[word] = d[word] + 1
    return d

class TextRankSummarizer(AbstractSummarizer):
    """An implementation of TextRank algorithm for summarization.

    Source: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
    """
    epsilon = 1e-4
    damping = 0.85
    # small number to prevent zero-division error, see https://github.com/miso-belica/sumy/issues/112
    _delta = 1e-7
    _stop_words = frozenset()

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count, model, idf):
        self._ensure_dependencies_installed()
        if not document.sentences:
            return ()

        ratings = self.rate_sentences(document, model, idf)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    @staticmethod
    def _ensure_dependencies_installed():
        if numpy is None:
            raise ValueError("LexRank summarizer requires NumPy. Please, install it by command 'pip install numpy'.")

    def rate_sentences(self, document, model, idf):
        matrix = self._create_matrix(document, model, idf)
        ranks = self.power_method(matrix, self.epsilon)
        return {sent: rank for sent, rank in zip(document.sentences, ranks)}

    def _create_matrix(self, document, model, idf):
        """Create a stochastic matrix for TextRank.

        Element at row i and column j of the matrix corresponds to the similarity of sentence i
        and j, where the similarity is computed as the number of common words between them, divided
        by their sum of logarithm of their lengths. After such matrix is created, it is turned into
        a stochastic matrix by normalizing over columns i.e. making the columns sum to one. TextRank
        uses PageRank algorithm with damping, so a damping factor is incorporated as explained in
        TextRank's paper. The resulting matrix is a stochastic matrix ready for power method.
        """
        sentences_as_words = [self._to_words_set(sent) for sent in document.sentences]
        sentences_count = len(sentences_as_words)
        weights = numpy.zeros((sentences_count, sentences_count))

        for i, words_i in enumerate(sentences_as_words):
            for j, words_j in enumerate(sentences_as_words):
                if model is None:
                    weights[i, j] = self._rate_sentences_edge(words_i, words_j)
                elif idf is None:
                    weights[i, j] = self._rate_sentences_edge_2(words_i, words_j, model)
                else:
                    weights[i, j] = self._rate_sentences_edge_3(words_i, words_j, model, idf)
        weights /= (weights.sum(axis=1)[:, numpy.newaxis]+self._delta) # delta added to prevent zero-division error
        #(see issue https://github.com/miso-belica/sumy/issues/112 )

        # In the original paper, the probability of randomly moving to any of the vertices
        # is NOT divided by the number of vertices. Here we do divide it so that the power
        # method works; without this division, the stationary probability blows up. This
        # should not affect the ranking of the vertices so we can use the resulting stationary
        # probability as is without any postprocessing.
        return numpy.full((sentences_count, sentences_count), (1.-self.damping) / sentences_count) \
            + self.damping * weights

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, sentence.words)
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    @staticmethod
    def _rate_sentences_edge(words1, words2):
        rank = 0
        for w1 in words1:
            for w2 in words2:
                rank += int(w1 == w2)

        if rank == 0:
            return 0.0

        assert len(words1) > 0 and len(words2) > 0
        norm = math.log(len(words1)) + math.log(len(words2))
        if numpy.isclose(norm, 0.):
            # This should only happen when words1 and words2 only have a single word.
            # Thus, rank can only be 0 or 1.
            assert rank in (0, 1)
            return rank * 1.0
        else:
            return rank / norm

    @staticmethod
    def _rate_sentences_edge_2(words1, words2, model):
        size = model['公安局'].size
        s1 = np.array([0]*size)
        for word in words1:
            if word not in model:
                tmp = np.array([0]*size)
            else:
                tmp = model[word]
            s1 = s1 + tmp
        if len(words1) == 0:
            s1 = np.array([0]*size)
        else:
            s1 = s1 / len(words1)

        s2 = np.array([0] * size)
        for word in words2:
            if word not in model:
                tmp = np.array([0]*size)
            else:
                tmp = model[word]
            s2 = s2 + tmp
        if len(words2) == 0:
            s2 = np.array([0]*size)
        else:
            s2 = s2 / len(words2)

        tmp = cosine_similarity(s1.reshape(1, -1), s2.reshape(1, -1))
        tmp = max(0, tmp)
        return tmp

    @staticmethod
    def _rate_sentences_edge_3(words1, words2, model, idf):
        size = model['公安局'].size

        words1_tf = count_tf(words1)
        s1 = np.array([0]*size)
        for word in words1:
            if word not in model:
                tmp = np.array([0] * size)
            else:
                tmp = model[word]
            if not word in idf:
                idf_word1 = 0
            else:
                idf_word1 = idf[word]
            tfidf = words1_tf[word] / len(words1) * math.log(100000 / (idf_word1 + 1))
            s1 = s1 + tmp * tfidf

        words2_tf = count_tf(words2)
        s2 = np.array([0] * size)
        for word in words2:
            if word not in model:
                tmp = np.array([0] * size)
            else:
                tmp = model[word]

            if not word in idf:
                idf_word2 = 0
            else:
                idf_word2 = idf[word]
            tfidf = words2_tf[word] / len(words2) * math.log(100000 / (idf_word2+1))
            s2 = s2 + tmp * tfidf

        tmp = cosine_similarity(s1.reshape(1, -1), s2.reshape(1, -1))
        tmp = max(0, tmp)
        return tmp

    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector
