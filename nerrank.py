from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
import pkuseg
import math
LANGUAGE = "chinese"
SENTENCES_COUNT = 3

if __name__ == "__main__":
    seg = pkuseg.pkuseg(postag=True)

    for i, line in enumerate(open("data/content_new_2.txt").readlines()):
        parser = HtmlParser.from_string(line, tokenizer=Tokenizer(LANGUAGE), url=None)
        sent_score = []
        for j, sent in enumerate(parser.document.sentences):
            text = sent._text
            segtext = seg.cut(text)
            nnum = len(list(filter(lambda x:x[1] == 'n', segtext)))
            rnum = len(list(filter(lambda x: x[0] == '本院', segtext)))
            r2num = len(list(filter(lambda x: x[0] == '认为', segtext)))
            lnum = math.log(len(segtext), 10)
            sent_score.append((text, (nnum+rnum*7+r2num*5) *1.0 / len(segtext) + lnum, j, segtext))

        num_sentences = int(max(min(5, len(parser.document.sentences) / 15), 3))
        sent_score.sort(key=lambda x:x[1], reverse=True)
        sent_idx = [(text, idx) for text, radio, idx, segtext in sent_score][:num_sentences]
        sent_idx.sort(key=lambda x:x[1])
        x = 0

        print(i, '原文')
        print(line)
        print('sentences:', len(parser.document.sentences))
        print()
        print('摘要：')
        for sentence, j in sent_idx:
            print('(%d)' % j, sentence)
        print()