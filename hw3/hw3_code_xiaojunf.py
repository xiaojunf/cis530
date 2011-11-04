import nltk
#from stanford_parser.parser import Parser
import os
from operator import itemgetter
def load_topic_words(topic_file):
    return dict([(line.split()) for
                    line in open(topic_file).readlines()])

def get_topic_n_words(topic_words_dict,n):
    return sorted(topic_words_dict,
                  key=topic_words_dict.__getitem__, reverse=True)[:n]

def filter_top_n_words(topic_words_dict,n, word_list):
    unique_words = list(set(word_list).intersection(
                                set(topic_words_dict.keys())))
    
    n = max(n, len(unique_words))
    return sorted(unique_words, topic_words_dict.__getitem__,
                                                reverse=True)[:n]

def load_file_sentences(path):
    file = open(path)
    text = file.read().replace('\n',' ').lower()
    file.close()
    return nltk.sent_tokenize(text)

def load_collection_sentences(path,n=None):
    listing = os.listdir(path)
    n=len(listing) if n==None else n
    sentences = [load_file_sentences(os.path.join(path,file))
                                    for file in listing[:n]]
    return reduce(lambda x,y:x+y,sentences)

def dependency_parse_sentence(sentence):
    parser = Parser()
    deps = parser.parseToStanfordDependencies(sentence)
    return [(r,gov.text,dep.text) for r,gov,dep in deps.dependencies]

def dependency_parse_collection(path):
    deps_list = [dependency_parse_sentence(s) for s
                        in load_collection_sentences(path)]
    return reduce(lambda x,y:x+y, deps_list)

def get_linked_words(dependency_list,word):
    def get_word((r,w,v)):
        if w==word:
            return v
        if v==word:
            return w
    return list(set(map(get_word, dependency_list)))

def create_graphviz_file(edge_list,output_file):
    out = open(output_file,'w')
    out.write('graph G {\n')
    for (w,v) in edge_list:
        out.write('%s -- %s;\n' %(w,v))
    out.write('}')
    out.close()

def get_top_n_linked_words(topic_word_dict,dependency_list,n,word):
    word_list = get_linked_words(dependency_list,word)
    return filter_top_n_words(topic_word_dict,word_list,n)
     
def visualize_collection_topics(topic_file,collection_path,output_file):
    tw = load_topic_words(topic_file)
    top_10_tw = get_top_n_words(tw,10) #TODO, SHOULD parse every sentence into word list?
    dps = dependency_parse_collection(collection_path)
    edge_list = [(w,v) for v in get_top_n_linked_words(dps,5,w) for w in top_10_tw]
    create_graphviz_file(edge_list,output_file)

def create_collection_feature_space(collection_path):
    sentence_list = load_collection_sentences(collection_path)
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

def vectorize_collection(feature_space,collection_path):
    sentence_list = load_collection_sentences(collection_path)
    return [(s,vectorize(feature_space,s)) for s in sentence_list]

def rank_by_centrality(collection_path,sim_func):
    fs = create_collection_feature_space(collection_path)
    vc = vectorize_collection(fs,collection_path)
    for i in range(len(vc)):
        vc[i][1] = sum([sim_func(vc[i][1],vc[j][1])
                    for j in len(vc) if i!=j])/(len(vc)-1)
    return sorted(vc,key=itemgetter(1),reverse=True)

def rank_by_tweight(collection_path,topic_file):
    tw = load_collection_sentences(topic_file)
    sent_list = load_collection_sentences(collection_path)
    def count_tw(sent):
        return len(set(sent.split()).intersection(
                                        set(tw.keys())))
    return sorted([(s, count_tw(s)) for s in sent_list],
                        key=itemgetter(1),reverse=True)

def summarize_ranked_sentences(ranked_sents,summary_len):
    result = []
    count = 0
    for sent in ranked_sents:
        if count+len(sent.split()) > summary_len:
            break
        result.append(sent)
        count += len(sent)
    return result

if __name__ =='__main__':
#    print load_topic_words('./data/topics.ts')
#    path = '/home1/c/cis530/data-hw3/articles/d30006t/'
#    print load_collection_sentences(path)
    edges = [('dog','cat'),('dog','computer'),('cat','computer')]
    create_graphviz_file(edges,'./data/test.gr')
    pass