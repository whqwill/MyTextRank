import pickle

f = open("content_s_seg.txt")

d = {}
for i, line in enumerate(f.readlines()):
    if i % 500 == 0:
        print (i)
    tmp = {}
    for w in line.split():
        if not w in tmp:
            tmp[w] = 1
            if not w in d:
                d[w] = 1
            else:
                d[w] = d[w] + 1
    x = 0

f = open('n-gram.pickle', 'wb')
pickle.dump(d, f)