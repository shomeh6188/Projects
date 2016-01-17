import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = nltk.WordNetLemmatizer()
STOPWORDS = [lemmatizer.lemmatize(t) for t in stopwords.words('english')]

def findcomplNominal(tag_sent):
    nom = []
    for i,(x,t) in enumerate(tag_sent):
        if tag_sent[i][1] == 'NN' and tag_sent[i-1][1] == 'JJ':
            nom.append(tag_sent[i-1][0] + " " + tag_sent[i][0])
            tag_sent.remove(tag_sent[i-1])
            tag_sent.remove((x,t))
    return nom

def findotherNominal(tag_sent):
    nom = []
    for i,(x,t) in enumerate(tag_sent):
        if tag_sent[i][1] == 'NN' and tag_sent[i-1][1] == 'NN':
            nom.append(tag_sent[i-1][0] + " " + tag_sent[i][0])
            tag_sent.remove(tag_sent[i-1])
            tag_sent.remove((x,t))
    return nom

def findnounAdj(tag_sent):
    na = []
    i = [tag_sent.index(x) for x in tag_sent if x[1] == "JJ"]
    for z in i:
        noun = [y[0] for y in tag_sent if tag_sent.index(y) > z and y[1] == "NN" or y[1] == "NNS"]
        na.append(tag_sent[z][0] + " " + ' '.join(map(str, noun)))
    return na

def findallNoun(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "NN" or x[1] == "NNS"]
    return i

def findallVerb(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "VBZ" or x[1] == "VB" or x[1] == "VBD" or x[1] == "VBG" or x[1] == "VBN" or x[1] == "VBP"]
    return i

def findallAdverb(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "RB" or x[1] == "RBR" or x[1] == "RBS"]
    return i

def countexp_nnp(para, nnp):
    cnt = 0
    for stxt in nnp:
        if stxt.lower() in para.lower():
            cnt = cnt + 1
            break
        else:
            for w in stxt.split():
                if w.lower() in para.lower():
                    cnt = cnt + 1
                    break
    return cnt

def countexp(para, searchText):
    cnt = 0
    for stxt in searchText:
        if stxt.lower() in para.lower():
            cnt = cnt + 1
    return cnt


def countexp_verb(para, searchText):
    cnt = 0
    searchWords = [WordNetLemmatizer().lemmatize(s.lower(), 'v') for s in searchText]
    for stxt in searchWords:
        for word in para.split():
            w1 = WordNetLemmatizer().lemmatize(word.lower(), 'v')
            if stxt.lower() == w1.lower():
                cnt = cnt + 1
                break
    return cnt

def countexp_noun(para, searchText):
    cnt = 0
    searchWords = [WordNetLemmatizer().lemmatize(s.lower(), 'n') for s in searchText]
    for stxt in searchWords:
        for word in para.split():
            w1 = WordNetLemmatizer().lemmatize(word.lower(), 'n')
            if stxt.lower() == w1.lower():
                cnt = cnt + 1
                break
    return cnt

def countexp_na(para, searchText):
    cnt = 0
    flag = 0
    for stxt in searchText:
        for sen in nltk.sent_tokenize(para):
            for word in stxt.split():
                if word in sen:
                    flag = 1
                else:
                    flag = 0
            if flag == 1:
                cnt = cnt + 1
                break
    return cnt