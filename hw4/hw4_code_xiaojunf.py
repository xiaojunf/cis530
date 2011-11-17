from nltk.corpus import wordnet as wn
from operator import itemgetter
import nltk
import os
def get_most_polysemous(n,word_list,part_of_speech):
    pos_dict = {
            'noun':'n',
            'verb':'v',
            'adjective':'a',
            'adverb':'r'
    }
    word_list = filter(lambda x:x[1]>0,
                [(w,len(wn.synsets(w,pos_dict[part_of_speech])))
                 for w in word_list])
    word_list = sorted(word_list, key=itemgetter(1),reverse=True)
    return [w for (w,n) in word_list[:n]]

def get_least_polysemous(n,word_list,part_of_speech):
    pos_dict = {
            'noun':'n',
            'verb':'v',
            'adjective':'a',
            'adverb':'r'
    }
    word_list = filter(lambda x:x[1]>0,
                [(w,len(wn.synsets(w,pos_dict[part_of_speech])))
                 for w in word_list])
    word_list = sorted(word_list, key=itemgetter(1))
    return [w for (w,n) in word_list[:n]]

def get_most_specific(n,word_list):
    word_list = [(w, min([synset.min_depth()
            for synset in wn.synsets(w,'n')]))
            for w in word_list if len(wn.synsets(w,'n'))>0]

    return [w for (w,n) in sorted(filter(lambda pair:pair[1]>0,word_list),
                        key=itemgetter(1),reverse=True)[:n]]

def get_least_specific(n,word_list):
    word_list = [(w, min([synset.min_depth()
            for synset in wn.synsets(w,'n')]))
            for w in word_list if len(wn.synsets(w,'n'))>0]

    return [w for (w,n) in sorted(filter(lambda pair:pair[1]>0,word_list),
                                    key=itemgetter(1))[:n]]

def get_similarity(word1,word2):
    pairs = [(x,y) for x in wn.synsets(word1,'n')
                   for y in wn.synsets(word2,'n')]
    if len(pairs)==0:
        return -1
    return max([x.path_similarity(y) for (x,y) in pairs])

def get_all_pairs_similarity(word_list):
    pairs = [(word_list[i],word_list[j])
            for i in range(len(word_list))
            for j in range(i+1,len(word_list))]

    return [(x,y,get_similarity(x,y)) for (x,y) in pairs]

def filter_pairs_similarity(pair_list,minimum):
    return filter(lambda item:item[2]>=minimum,pair_list)

def get_similar_groups(word_list,minimum):
    pairs = filter_pairs_similarity(
            get_all_pairs_similarity(word_list),minimum)
    neighbor_dict = nltk.defaultdict(set)
    def add2dict(item):
        neighbor_dict[item[0]].add(item[1])
        neighbor_dict[item[1]].add(item[0])
    map(add2dict,pairs)
    result=[]
    def BronKerbosch(R,P,X):
        if len(P)==0 and len(X)==0:
            result.append(list(R))
            return 
        u = P.union(X).pop()
        for v in P.difference(neighbor_dict[u]):
            BronKerbosch(R.union(set([v])),
                P.intersection(neighbor_dict[v]),
                X.intersection(neighbor_dict[v]))
            P = P.difference(set([v]))
            X = X.union(set([v]))
    BronKerbosch(set([]),set(neighbor_dict.keys()),set([]))
    return [c for c in result if len(c)>2]


def load_collection_words(path):
    listing = os.listdir(path)
    word_list = [nltk.word_tokenize(s) for file in listing
                 for s in open(os.path.join(path,file)).readlines()]
    return list(set(reduce(lambda x,y:x+y, word_list)))

def load_topic_words(path):
    cin = open(path)
    tw = [line.split()[0] for line in cin.readlines()]
    cin.close()
    return tw

if __name__=='__main__':
    path = './data/'
    listing = os.listdir(path)
    for l in listing:
        print l
        tw = load_topic_words(os.path.join(path,l))
        print get_similar_groups(tw,0.25)

        
#    print tw[:5]
#    pos_list = ['noun','verb','adjective','adverb']
#    for pos in pos_list:
#        print pos
#        print get_most_polysemous(10,tw,pos)
#    wl = load_collection_words('/home1/c/cis530/data-hw3/articles/d31013t/')
#    print wl[:5]
#    print get_most_specific(10,tw)
#    print get_least_specific(10,tw)
#    print get_least_specific(10,tw)
        

        

#    pairs = [(1,2,1),(1,8,1),(2,8,1),(1,3,1),(3,5,1),
#                (5,7,1),(2,7,1),(3,4,1),(4,6,1),(5,6,1),
#                (3,6,1),(4,5,1)]
#    neighbor_dict = nltk.defaultdict(set)
#    def add2dict(item):
#        neighbor_dict[item[0]].add(item[1])
#        neighbor_dict[item[1]].add(item[0])
#    map(add2dict,pairs)
#    result=[]
#    def BronKerbosch(R,P,X):
#        if len(P)==0 and len(X)==0:
#            result.append(list(R))
#            return
#        u= P.union(X).pop()
#        for v in P.difference(neighbor_dict[u]):
#            BronKerbosch(R.union(set([v])),
#                P.intersection(neighbor_dict[v]),
#                X.intersection(neighbor_dict[v]))
#            P = P.difference(set([v]))
#            X = X.union(set([v]))
#    BronKerbosch(set([]),set(neighbor_dict.keys()),set([]))
#    print [c for c in result if len(c)>2]
    pass