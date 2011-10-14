__author__ = 'xiaojun'

from nltk.corpus import PlaintextCorpusReader
import nltk
from nltk.probability import FreqDist
import os
import math

ROOT = '/home1/c/cis530/data-hw2/'

def __output_header(output, dataset,fileid):
    output.write(dataset+'/'+fileid+' ')
    output.write(fileid.split('/')[0]+' ')

def __write_coarse(output,files,fileid):
    stop_list = open(ROOT+'stopwlist.txt').read().split()
    tokens = files.raw(fileid).split()
    sents = files.raw(fileid).split('\n')[:-1]
    output.write('tok:%d ' %len(tokens))
    output.write('typ:%d ' %len(set(tokens)))
    output.write('con:%d ' %len([x for x in tokens if x not in stop_list]))
    output.write('sen:%d ' %len(sents))
    output.write('len:%0.2f ' %(float(len(tokens))/float(len(sents))))
    output.write('cap:%d ' %len([w for w in tokens if w[0]<='Z' and w[0]>='A']))

def get_coarse_level_features(dataset,output_file):

    output = open(output_file,'w')
    root = ROOT+dataset

    files = PlaintextCorpusReader(root,'.*')
    for fileid in files.fileids():
        __output_header(output,dataset,fileid)
        __write_coarse(output,files,fileid)
        output.write('\n')
    output.close()

def __merge_tag():
    NOUNS = ['NN','NNS','NP','NPS']
    VERBS = ['VB','VBD','VBG','VBN','VBP','VBZ']
    ADJ = ['JJ','JJR','JJS']
    ADV = ['RB','RBR','RBS']
    PREP = ['IN']
#    tag_set = set(NOUNS).union(set(VERBS)).union(set(ADJ)).union(set(ADV)).union(set(PREP))
    tag_set = NOUNS+VERBS+ADJ+ADV+PREP
    TAG_DIC = {}
    def add2dic(ls, tag):
        for l in ls:
            TAG_DIC[l]=tag
    add2dic(NOUNS,'NN')
    add2dic(VERBS,'VV')
    add2dic(ADJ,'ADJ')
    add2dic(ADV,'ADV')
    add2dic(PREP,'PREP')
    return tag_set,TAG_DIC

def prepare_pos_features(Language_model_set, output_file):
    tag_set,TAG_DIC = __merge_tag()
    
    root = ROOT+Language_model_set
    files = PlaintextCorpusReader(root,'.*')
    pos_tag_list = nltk.pos_tag(files.raw().split())
    cfd = nltk.ConditionalFreqDist((TAG_DIC[t],w) for (w,t) in pos_tag_list if t in tag_set)

    output = open(output_file,'w')
    def write2file(tag,num):
        for w in cfd[tag].keys()[:num]:
            output.write(tag+w+'\n')
    write2file('NN',200)
    write2file('VV',200)
    write2file('ADJ',200)
    write2file('ADV',100)
    write2file('PREP',100)
    output.close()

def __write_pos(output,files,fileid,feature_list):
    tag_set,TAG_DIC = __merge_tag()
    pos_tag_list = nltk.pos_tag(files.raw(fileid).split())
    word_list = [TAG_DIC[t]+w for (w,t) in pos_tag_list if t in tag_set]
    fqd = FreqDist(word_list)
    for w in feature_list:
            output.write('%s:%d '%(w,fqd[w]))

def get_pos_features(dataset,feature_set_file,output_file):
    root = ROOT+dataset
    files = PlaintextCorpusReader(root,'.*')
    feature_list = open(feature_set_file).read().split()

    output = open(output_file,'w')
    for fileid in files.fileids():
        __output_header(output,dataset,fileid)
        __write_pos(output,files,fileid,feature_list)
        output.write('\n')
    output.close()

class BigramModel:
    def __init__(self,root,category):
        self.word_list = PlaintextCorpusReader(
            os.path.join(root,category),'.*').raw().split()
        self.bigram = nltk.bigrams(self.word_list)
        self.freqdist = FreqDist(self.word_list)
        self.cfd = nltk.ConditionalFreqDist(self.bigram)
    def get_prob_per(self,word_list): #TODO MODIFY THE FIRST WORD FREQUENCY
        N = len(set(word_list))
        prob = math.log(self.freqdist.freq(word_list[0]))
        for (pre,w) in nltk.bigrams(word_list):
            prob = prob+math.log(self.cfd[pre][w]+1)-math.log(
                        len(self.cfd[pre].keys())+N)
        return prob, -len(word_list)*prob

def __write_lm(output,files,fileid,fin_model,hel_model,res_model,co_model):

    word_list = files.raw(fileid).split()
    finprob,finper = fin_model.get_prob_per(word_list)
    hlprob,hlper = hel_model.get_prob_per(word_list)
    resprob,resper = res_model.get_prob_per(word_list)
    coprob,coper = co_model.get_prob_per(word_list)
    output.write('finprob:%0.1f hlprob:%0.1f resprob:%0.1f coprob:%0.1f'
                                    %(finprob,hlprob,resprob,coprob))
    output.write('finper:%0.1f hlper:%0.1f resper:%0.1f coper:%0.1f'
                                    %(finper,hlper,resper,coper))



def get_lm_features(dataset,output_file):
    root = ROOT+dataset
    files = PlaintextCorpusReader(root,'.*')
    fin_model = BigramModel('Finance',root)
    hel_model = BigramModel('Health',root)
    res_model = BigramModel('Computer_and_the_Internet',root)
    co_model = BigramModel('Research',root)
    output = open(output_file,'w')
    for fileid in files.fileids():
        __output_header(output,dataset,fileid)
        __write_lm(output,files,fileid,
                   fin_model,hel_model,res_model,co_model)
        output.write('\n')
    output.close()

def get_feature_file(directory_name,features_to_use,output_file):
    root = ROOT+directory_name
    files = PlaintextCorpusReader(root,'.*')
    fin_model = None
    hel_model = None
    res_model = None
    co_model = None
    
    output = open(output_file,'w')
    feature_list = open(feature_set_file).read().split()# TODO, IF POS, WHERE IS FEATURE LIST
    if features_to_use.__contains__('lm'):
        fin_model = BigramModel('Finance',root)
        hel_model = BigramModel('Health',root)
        res_model = BigramModel('Computer_and_the_Internet',root)
        co_model = BigramModel('Research',root)

    for fileid in files.fileids():
        __output_header(output,directory_name,fileid)
        if features_to_use.__contains__('lm'):
            __write_lm(output,files,fileid,
                   fin_model,hel_model,res_model,co_model)
        if features_to_use.__contains__('pos'):
            __write_pos(output,files,fileid,feature_list)
        if features_to_use.__contains__('coarse'):
            __write_coarse(output,files,fileid)
        output.write('\n')
    output.close()

if __name__ == '__main__':
#    get_coarse_level_features('Training_set_sm
# all','test')
    prepare_pos_features('Training_set_small', 'taggs')
