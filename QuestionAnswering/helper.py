import nltk
import constants
import copy
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = nltk.WordNetLemmatizer()
STOPWORDS = [lemmatizer.lemmatize(t) for t in stopwords.words('english')]
#from nltk.tag.stanford import StanfordNERTagger
#st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
year_list = []
for i in range(1400,2000):
    year_list.append(str(i))

def convertSenttolist(s):
    sent_li = []
    for word in s.split():
        sent_li.append(word)
    return sent_li

def findinList(L,lt):
    li = []
    for q in L.split():
        if q in lt:
            li.append(q)
    return li

def convertListtoSent(list):
    sent = ""
    for li in list:
        sent = sent + li + " "
    return sent

def findWH(sent):
    for word in reversed(sent.split()):
        if constants.ques_words.__contains__(word.lower()):
            return word
    return ""

def Postagger(sent):
    sent_tag = nltk.pos_tag(nltk.word_tokenize(sent))
    return sent_tag

def findNNP(sent_tagged):
    #i = [x for x in sent_tagged if x[1] == "NNP" or x[1] == "NNPS"]
    nnp = []
    flag = 0
    s = None
    for x in sent_tagged:
        if x[1] == "NNP" and flag == 0:
            s = x[0]
            flag = 1
        elif x[1] == "NNP" and flag == 1:
            s = s + " " + x[0]
        else:
            if s != None:
                nnp.append(s)
            s = None
            flag = 0
    if sent_tagged[-1][1] == "NNP":
        nnp.append(s)
    #for x in i:
        #sent_tagged.remove(x)
    return nnp

def removestopwords_punct(ques):
    t_ques = copy.deepcopy(ques)
    for w in t_ques:
        word_lm = lemmatizer.lemmatize(w[0].lower())
        if word_lm in STOPWORDS or word_lm in string.punctuation:
            ques.remove(w)
    return ques

def removepunc(val):
    val = val.replace("-"," ")
    val = val.replace(",","")
    val = val.replace("(", "")
    val = val.replace(")", "")
    val = val.replace("\\", "")
    val = val.replace("\"","")
    val = val.replace("!","")
    return val

def remqueswords(q,val):
    if q.lower().__contains__(val.lower()):
        return True
    return False

def removezeroscore(sent):
    temp = []
    for s in sent:
        if not s[1] == 0:
            temp.append(s)
    return temp

def findverbsinques(Whword, sent):
    verb_list  = []
    sent = sent.split(Whword)[1]
    sent_tag = Postagger(sent)
    sent_tag_nosw = removestopwords_punct(sent_tag)
    for sen in sent_tag_nosw:
        if sen[1] == "VB" or sen[1] == "VBD" or sen[1] == "VBG" or sen[1] == "VBN" or sen[1] == "VBP":
            verb_list.append(sen[0])
            break
    return verb_list

def remove_puncts(sen):
    sen = sen.replace(",","")
    sen = sen.replace("-", " ")
    sen = sen.replace("(", "")
    sen = sen.replace(")", "")
    sen = sen.replace("\\", "")
    sen = sen.replace("\"","")
    sen = sen.replace("!","")
    return sen