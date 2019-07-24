# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
LANGUAGE = "chinese"
SENTENCES_COUNT = 3

def randomEmbedding(size):
    return np.array([(np.random.random()-0.5)/size for i in range(size)])

def bestSentences(document, SENTENCES_COUNT, model):
    words = list(document.words)
    sentences = [list(sent.words) for sent in document.sentences]

    size = model['公安局'].size
    s1 = np.array([0] * size)
    for word in words:
        if word not in model:
            tmp = randomEmbedding(size)
        else:
            tmp = model[word]
        s1 = s1 + tmp
    if len(words) == 0:
        s1 = randomEmbedding(size)
    else:
        s1 = s1 / len(words)

    sent_scores = []

    for i, words2sent in enumerate(zip(sentences, document.sentences)):
        words2, sent = words2sent
        s2 = np.array([0] * size)
        for word in words2:
            if word not in model:
                tmp = randomEmbedding(size)
            else:
                tmp = model[word]
            s2 = s2 + tmp
        if len(words2) == 0:
            s2 = randomEmbedding(size)
        else:
            s2 = s2 / len(words2)

        score = cosine_similarity(s1.reshape(1, -1), s2.reshape(1, -1))
        sent_scores.append((sent._text, score[0][0], i))
    sent_scores.sort(key=lambda x:x[1], reverse=True)
    sents = [(sent, i) for sent, score, i in sent_scores][:SENTENCES_COUNT]
    sents.sort(key=lambda x:x[1])
    return [sent for sent, i in sents]

def bestSentences2(document, SENTENCES_COUNT):
    words = list(document.words)
    sentences = [list(sent.words) for sent in document.sentences]

    sent_scores = []

    for i, words2sent in enumerate(zip(sentences, document.sentences)):
        words2, sent = words2sent

        rank = 0
        for w1 in words:
            for w2 in words2:
                rank += int(w1 == w2)

        if rank == 0:
            score = 0.0
        else:
            norm = math.log(len(words)) + math.log(len(words2))
            if np.isclose(norm, 0.):
                score =  rank * 1.0
            else:
                score =  rank / norm

        sent_scores.append((sent._text, score, i))

    sent_scores.sort(key=lambda x:x[1], reverse=True)
    sents = [(sent, i) for sent, score, i in sent_scores][:SENTENCES_COUNT]
    sents.sort(key=lambda x:x[1])
    return [sent for sent, i in sents]

if __name__ == "__main__":
    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    model = gensim.models.Word2Vec.load('model/MyModel')

    for i, line in enumerate(open("data/content_new_2.txt").readlines()):
        parser = HtmlParser.from_string(line, tokenizer=Tokenizer(LANGUAGE), url=None)

        print (i, '原文')
        print (line)
        print ()
        print ('sentences:', len(parser.document.sentences))
        print ()

        num_sentences = int(max(min(5, len(parser.document.sentences)/15), 3))

        print ('摘要1：')
        for j, sentence in enumerate(summarizer(parser.document, num_sentences, model)):
            print('(%d)' %j, sentence)
        print ()

        print('摘要2：')
        for j, sentence in enumerate(summarizer(parser.document, num_sentences, None)):
            print('(%d)' %j, sentence)
        print()

        print('摘要3：')
        for j, sentence in enumerate(bestSentences(parser.document, num_sentences, model)):
            print('(%d)' %j, sentence)
        print()

        print('摘要4：')
        for j, sentence in enumerate(bestSentences2(parser.document, num_sentences)):
            print('(%d)' %j, sentence)
        print()

        print ()
        print ()
        print ()