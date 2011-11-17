from nltk.corpus import wordnet as wn
syn_words={}
ant_words={}
def get_syn_ant_words(word_tag):
    try:
        return syn_words[word_tag],ant_words[word_tag]
    except KeyError:
        w,i = word_tag.split('_')
        syn_sets = wn.synsets(w,i)
        syn_words[word_tag]=set([lemma.name+'_'+i
                    for synset in syn_sets
                    for lemma in synset.lemmas ])
        ant_words[word_tag]=set([a_lemma.name+'_'+i
                            for synset in syn_sets
                            for lemma in synset.lemmas
                            for a_lemma in lemma.antonyms()])
        return syn_words[word_tag],ant_words[word_tag]

def one_round(pre_words,lam):
    cur_words={}
    get = cur_words.get
    for word in pre_words.keys():
        syn,ant = get_syn_ant_words(word)
        for w in syn:
            factor = 1+lam if word==w else lam
            cur_words[w]=get(w,0)+pre_words[word]*factor
        for w in ant:
            factor = -lam
            cur_words[w]=get(w,0)+pre_words[word]*factor
    return cur_words

#def one_round(neg,pos,neu,lam):
#    neg_new = {}
#    pos_new = {}
#    neu_new = {}
#    for word in neg.keys():
#        syn,ant = get_syn_ant_words(word)
#        for w in syn:
#            factor = 1+lam if word==w else lam
#            try:
#                neg_new[w]=neg_new[w]+neg[word]*factor
#            except KeyError:
#                neg_new[w]=neg[word]*factor
#        for w in ant:
#            factor = -lam
#            try:
#




def calculate_sensitive_score(neg_file,pos_file,neu_file):
    N = [open(neg_file).read().split('\t')]
    P = [open(pos_file).read().split('\t')]
    M = [open(neu_file).read().split('\t')]

    cur_words = dict([(w,1) for w in N]+[(w,-1) for w in P])
#    neg_word = dict([(w,1) for w in N])
#    pos_word = dict([(w,-1) for w in P])
#    neu_word = dict([(w,-1) for w in M])

    lam=0.2

    for i in range(5):
        neg_word,pos_word,neu_word = one_round(cur_words,lam)







    
if __name__=='__main__':
    syn_words['a']=set(['a'])
    ant_words['a']=set(['d'])

    syn_words['b']=set(['b','c'])
    ant_words['b']=set([])

    syn_words['c']=set(['c','b'])
    ant_words['c']=set([])
    syn_words['d']=set(['e'])
    ant_words['d']=set(['a'])
    
    syn_words['e']=set(['d'])
    ant_words['e']=set([])
    cur_words = {'a':1,'b':-1}
    for i in range(3):
     cur_words = one_round(cur_words,0.2)
     print cur_words




  