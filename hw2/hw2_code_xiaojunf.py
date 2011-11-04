__author__ = 'xiaojun'

from nltk.corpus import PlaintextCorpusReader
import nltk
from nltk.probability import FreqDist
import os
import math
from operator import itemgetter

ROOT = '/home1/c/cis530/data-hw2/'

def __output_header(output, dataset,fileid):
    output.write(dataset+'/'+fileid+' ')
    output.write(fileid.split('/')[0]+' ')

def __write_coarse(output,files,fileid):
    stop_list = open(ROOT+'stopwlist.txt').read().split()
    tokens = files.words(fileid)
    sents = files.sents(fileid)
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
    pos_tag_list = nltk.pos_tag(files.words())
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
    pos_tag_list = nltk.pos_tag(files.words(fileid))
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
    def __init__(self,category,root):
        self.word_list = PlaintextCorpusReader(
            os.path.join(root,category),'.*').words()
        self.bigram = nltk.bigrams(self.word_list)
        self.freqdist = FreqDist(self.word_list)
        self.cfd = nltk.ConditionalFreqDist(self.bigram)
    def get_prob_per(self,word_list): #TODO MODIFY THE FIRST WORD FREQUENCY
        N = len(set(word_list))

#        prob = math.log(self.freqdist[word_list[0]]+1)-math.log(len(self.freqdist.keys())+N)
        prob = 0
        for (pre,w) in nltk.bigrams(word_list):
            prob = prob+math.log(self.cfd[pre][w]+1)-math.log(
                        len(self.cfd[pre].keys())+N)
        return prob, -prob/float(len(word_list))

def __write_lm(output,files,fileid,fin_model,hel_model,res_model,co_model):

    word_list = files.words(fileid)
    finprob,finper = fin_model.get_prob_per(word_list)
    hlprob,hlper = hel_model.get_prob_per(word_list)
    resprob,resper = res_model.get_prob_per(word_list)
    coprob,coper = co_model.get_prob_per(word_list)
    output.write('finprob:%0.1f hlprob:%0.1f resprob:%0.1f coprob:%0.1f '
                                    %(finprob,hlprob,resprob,coprob))
    output.write('finper:%0.1f hlper:%0.1f resper:%0.1f coper:%0.1f'
                                    %(finper,hlper,resper,coper))



def get_lm_features(dataset,output_file):
    root = ROOT+dataset
    files = PlaintextCorpusReader(root,'.*')
    fin_model = BigramModel('Finance',root)
    hel_model = BigramModel('Health',root)
    res_model = BigramModel('Computers_and_the_Internet',root)
    co_model = BigramModel('Research',root)
    output = open(output_file,'w')
    for fileid in files.fileids():
        word_list = files.words(fileid)
        finprob,finper = fin_model.get_prob_per(word_list)
        hlprob,hlper = hel_model.get_prob_per(word_list)
        resprob,resper = res_model.get_prob_per(word_list)
        coprob,coper = co_model.get_prob_per(word_list)
        __output_header(output,dataset,fileid)
        output.write('finprob:%0.1f hlprob:%0.1f resprob:%0.1f coprob:%0.1f '
                                    %(finprob,hlprob,resprob,coprob))
        output.write('finper:%0.1f hlper:%0.1f resper:%0.1f coper:%0.1f'
                                    %(finper,hlper,resper,coper))
#        __write_lm(output,files,fileid,
#                   fin_model,hel_model,res_model,co_model)
        output.write('\n')
    output.close()

def combine_features(feature_files_list,output_file):
    fins = [dict(line.split(None,1) for line in open(file).readlines()) for file in feature_files_list]
    fout = open(output_file,'w')
    for (k,v) in fins[0].items():
        fout.write(k+' '+v.split(None,1)[0])
        for file_dic in fins:
            fout.write(' '+file_dic[k].split(None,1)[1][:-1])
        fout.write('\n')

    fout.close()

def __get_features(text): # return docID, featureset, label
    text = text.split(None,2)
    return [text[0],(dict(pair.split(':') for pair in text[2].split()),text[1])]

def get_NB_classifier(train_examples):
    train_sets = [__get_features(line)[1] for line in open(train_examples).readlines()]
    return nltk.NaiveBayesClassifier.train(train_sets)

def classify_documents(test_examples,model,classifier_output):
    output = open(classifier_output,'w')
    for line in open(test_examples).readlines():
        test_feature = __get_features(line)  #fileid, (featureset, category)
        output.write(test_feature[0]+' '+test_feature[1][1]+
                     ' '+model.classify(test_feature[1][0])+'\n')
    output.close()

def get_fit_for_word(sentence,word,langmodel):
    root = ROOT+'Language_model_set'
    model = BigramModel(langmodel,root)
    sentence = sentence.replace('<blank>', word)
    return model.get_prob_per(nltk.word_tokenize(sentence))[0]

def get_bestfit_topic(sentence, wordlist,topic):
    return max([(word,get_fit_for_word(sentence,word,topic))
        for word in wordlist],key=itemgetter(1))[0]

def writeup_1_6_1(test_file):
    true_label = []
    pred_label = []

    def add2list(line):
        line = line.split()
        true_label.append(line[1])
        pred_label.append(line[2])
    map(add2list, open(test_file).readlines())
    cm = nltk.ConfusionMatrix(true_label,pred_label)
    print cm.pp(sort_by_count=True, show_percents=True,truncate=9)
if __name__ == '__main__':
#    get_coarse_level_features('Training_set_small','Training_set_small.coarsefeatures')
#    prepare_pos_features('Language_model_set', 'taggs')
#    get_lm_features('Test_set','Test_set.lmfeatures')
#    get_lm_features('Training_set_small','Training_set_small.lmfeature')
#    prepare_pos_features('Language_model_set','feature_set_file')
#    combine_features(['Test_set.lmfeatures','Test_set.coarsefeatures','Test_set.posfeatures'],'Test_set.lmcoaposfeatures')
#    combine_features(['Training_set_small.coarsefeatures','Training_set_small.lmfeatures','Training_set_small.posfeatures'],'Training_set_small.coalmposfeatures')
#    combine_features(['Training_set_large.coarsefeatures','Training_set_large.posfeatures'],'Training_set_large.coaposfeatures')
#    combine_features(['Test_set.lmfeatures','Test_set.posfeatures'],'Test_set.lmposfeatures')
#    combine_features(['Test_set.coarsefeatures','Test_set.posfeatures'],'Test_set.coaposfeatures')
#    classifier = get_NB_classifier("Training_set_large.lmposfeatures")
#    classify_documents("Test_set.lmposfeatures",classifier,"test_set_lmpos.pred")
#    classifier = get_NB_classifier("Training_set_small.coalmposfeatures")
#    classify_documents("Test_set.lmcoaposfeatures",classifier,"test_set_small_lmcoapos.pred")

    writeup_1_6_1('test_set_small_lmcoapos.pred')


    pass