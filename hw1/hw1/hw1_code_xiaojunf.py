from nltk.corpus import brown
from nltk.probability import FreqDist
import nltk
import math
import numpy as np
import random

def get_prob_word_in_category(word, category=None):
    category_text = brown.words(categories = category)
    return category_text.count(word)*1.0/len(category_text)

def get_vocabulary_size(category=None):
    return len(set(brown.words(categories = category)))

def get_type_token_ratio(category=None):
    category_text = brown.words(categories = category)
    return len(set(category_text))/len(category_text)

def get_entropy(category=None):
    frq = FreqDist(brown.words(categories = category))
    return sum(map(lambda w: -frq.freq(w)*math.log(frq.freq(w),2), frq.keys()))



def get_top_n_words(n, category=None):
    return FreqDist(brown.words(categories = category)).keys()[:n]

def get_bottom_n_words(n,category=None):
    return FreqDist(brown.words(categories = category)).keys()[-n:]

def plot_word_counts(): #TODO
    import matplotlib.pyplot as plt
    word_count = FreqDist(brown.words(categories = 'news')).values()
    plt.hist(word_count,bins=3000)
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Number of Words')
    plt.title('Word Frequency')
    plt.axis([0,500,0,500])
    plt.show()


def get_word_contexts(word):
    text = brown.words(categories = 'news')
    return [(text[i-1],text[i+1]) for(i,w) in enumerate(list(text))
                                  if w==word and i>0 and i<len(text)-1]

def get_common_contexts(word1,word2):
    return list(set(get_word_contexts(word1)).intersection(
                                        set(get_word_contexts(word2))))

def create_feature_space(sentence_list):
    word_list = set(reduce(lambda x,y:x+y,
                           map(lambda x:x.split(),sentence_list)))
    return dict((w,i) for (i,w) in enumerate(word_list))

def vectorize(feature_space,sentence):
    vector = [0 for i in range(len(feature_space))]
    def add2vector(w):
        try:
            vector[feature_space[w]]=1
        except KeyError:
            pass
    map(add2vector, sentence.split())
    return vector

def jaccard_similarity(X,Y):
    return 2.0*sum(np.array(X)*np.array(Y))/(sum(X)+sum(Y))

def dice_similarity(X,Y):
    x_array = np.array(X)
    y_array = np.array(Y)
    return sum(x_array*y_array)*1.0/len((x_array+y_array).nonzero())

def cosine_similarity(X,Y):
    return sum(np.array(X)*np.array(Y))*1.0/(sum(X)*sum(Y))

def make_ngram_tuples(samples,n):
    def get_ngram(i):
        if n == 1:
            return (None,samples[i])
        return (tuple([samples[i-j] for j in range(n-1,0,-1)]),samples[i])
    return map(get_ngram,range(n-1,len(samples)))


class NGramModel:
    def __init__(self,training_data,n):
        self.__model = make_ngram_tuples(training_data,n)
        self.__cfd = nltk.ConditionalFreqDist(self.__model)

    def prob(self,context,event):
        return self.__cfd[context].freq(event)

    def generate(self,n,context):

        def add_remove2tuple(add, t):
            t = list(t)
            t.append(add)
            t.pop(0)
            return tuple(t)

        sentence = []
        if context:
            sentence.extend(context)

        for i in range(n-len(sentence)):
            random_p = random.random()
            sum_p = 0
            for w in self.__cfd[context].keys():
                sum_p = sum_p + self.__cfd[context].freq(w)
                if sum_p >= random_p:
                    sentence.append(w)
                    if context:
                        context = add_remove2tuple(w,context)
                    break
            if sum_p ==0:
                break
        return sentence



            



if __name__ == '__main__':

    sentences = ["the quick brown fox jumped over the lazy dog",
                         "the fox leaped over the tired dog",
                         "the fox",
                         "the lazy dog the lazy dog the lazy dog"]

    """
    This is code for write up 2.1(c)
    """
    def get_average_sharedContext():  #for 2.1(c)
        words = ["Washington","Philadelphia","Boston","London"]
        pair = [[(words[i],words[j])for j in range(i+1,4,1)] #generate 6 pairs of the words
                                        for i in range(3)]
        pair = pair[0]+pair[1]+pair[2]
        sum = 0
        for (w,v) in pair:                                  #calculate average value
            sum += len(get_common_contexts(w,v))
        return sum*1.0/len(pair)


    """
    This is the code for 2.2(c) to use the three metrics to calculate similarity
    Input is sentence list that is ready to use.
    Rank is a dictionary to store the similarity value of 6 pairs using 3 different metrics,
    The data stored in rank is like this {'dice':[(i, value),...,(j, value)]}
    The 'dice' means using the dice metric;,(i,value) means the similarity of the
    ith pair among the 6 is value. And Rank is sorted by the value reversely.

    """
    def calculate_similarity(sentences): #for 2.2(c)

        space = create_feature_space(sentences)
        vectors = map(lambda x: vectorize(space,x), sentences)
        pairs = [[[vectors[i],vectors[j]]for j in range(i+1,4,1)]
                                                for i in range(3)]
        pairs = pairs[0]+pairs[1]+pairs[2]
        rank = {}
        rank['jaccard']=[]
        rank['dice']=[]
        rank['cosine']=[]
        i = 0
        for (x,y) in pairs:
            rank['jaccard'].append((i,jaccard_similarity(x,y)))
            rank['dice'].append((i,dice_similarity(x,y)))
            rank['cosine'].append((i,cosine_similarity(x,y)))
            i+=1
        def reverse_sort(x):
            rank[x]=sorted(rank[x],key=lambda t:t[1],reverse=True)
        map(reverse_sort,rank.keys())
        return rank

    """
    This is code for 2.2(e)
    I mainly use tf_idf and consine to caculate the similarity

    get_idf_dic, returns a dictionary of document index, the key is word,
     and the value is a list of the number of sentences in which the word occurs.

    get_vector returns vector of a sentence with the weight using tf*idf

    get_similarity simply return the similarity using consine function
    """
    def tf_idf(sentence_list):

        def get_idf_dic(sentence_list):
            word_dict = {}
            for i in range(len(sentence_list)):
                for w in sentence_list[i].split():
                    try:
                        word_dict[w].add(i)
                    except:
                        word_dict[w]=set([i])
            return word_dict

        def get_vector(space,idf_dic,sentence):
            vector = [0 for i in range(len(space))]
            frq = FreqDist(sentence.split())
            def get_tf_idf(word):
                vector[space[word]]=frq.freq(word)*math.log(len(sentence_list)*1.0
                                        /len(idf_dic[word]),2)
            map(get_tf_idf,sentence.split())
            return vector

        def get_similarity(X,Y):
            return cosine_similarity(X,Y)

        space = create_feature_space(sentence_list)
        idf_dic = get_idf_dic(sentence_list)
        vectors = map(lambda x: get_vector(space,idf_dic,x), sentence_list)
        pairs = [[(vectors[i],vectors[j])for j in range(i+1,len(sentence_list),1)]
                                                for i in range(len(sentence_list)-1)]

        pairs = reduce(lambda x,y:x+y,pairs)
        return sorted([(i,get_similarity(pairs[i][0],pairs[i][1]))
                                        for i in range(len(pairs))],
                                        key=lambda p: p[1],reverse=True)

    print tf_idf(sentences)