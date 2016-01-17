import constants
import copy
import nltk
import string
from nltk.corpus import stopwords
lemmatizer = nltk.WordNetLemmatizer()
STOPWORDS = [lemmatizer.lemmatize(t) for t in stopwords.words('english')]
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.wordnet import WordNetLemmatizer
import constants
import helper

def remstwords(str):
    t_str = copy.deepcopy(str)
    for w in t_str:
        word_lm = lemmatizer.lemmatize(w[0].lower())
        if word_lm in STOPWORDS or string.punctuation:
            str.remove(w)
    return str

def checkStopWord(word):
    word_lm = lemmatizer.lemmatize(word.lower())
    if word_lm in STOPWORDS:
        return True
    return False

def PosTag(sent):
    sent_tag = nltk.pos_tag(nltk.word_tokenize(sent))
    return sent_tag

def WordMatch(ques, sent):
    score = 0
    ques_tag = PosTag(ques)
    sent_tag = PosTag(sent)
    for q in ques_tag:
        if not checkStopWord(q[0]):
            for s in sent_tag:
                if s[1] == "VB" or s[1] == "VBD" or s[1] == "VBG" or s[1] == "VBN" or s[1] == "VBP":
                    w1 = WordNetLemmatizer().lemmatize(q[0].lower(), 'v')
                    w2 = WordNetLemmatizer().lemmatize(s[0].lower(), 'v')
                    if w1 == w2:
                        score = score + 6
                        break
                else:
                    w = ps.stem(q[0].lower())
                    s_lm = ps.stem(s[0].lower())
                    if w == s_lm:
                        score = score + 3
                        break
    if containsNER(ques,"PERSON") or containsList(ques, constants.PERSON_NAMES):
        if containsNER(sent, "PERSON") or containsList(sent, constants.PERSON_NAMES):
            y=1
        else:
            for se in sent_tag:
                if se[1] == "PRP":
                    score = score + 3
                    break
    return score

def WordMatchHow(ques, sent):
    score = 0
    ques_tag = PosTag(ques)
    sent_tag = PosTag(sent)
    if containsNER(ques,"PERSON") or containsList(ques, constants.PERSON_NAMES):
        for se in sent_tag:
            if se[1] == "PRP":
                score = score + 3
                break
    for s in sent_tag:
        if not checkStopWord(s[0]):
            if s[1] == "VB" or s[1] == "VBD" or s[1] == "VBG" or s[1] == "VBN" or s[1] == "VBP":
                for q in ques_tag:
                    w1 = WordNetLemmatizer().lemmatize(q[0].lower(), 'v')
                    w2 = WordNetLemmatizer().lemmatize(s[0].lower(), 'v')
                    if w1 == w2:
                        score = score + 6
                        break
            else:
                w = ps.stem(s[0].lower())
                for q in ques_tag:
                    s_lm = ps.stem(q[0].lower())
                    if w == s_lm:
                        score = score + 3
                        break
    return score

def containsNER(q,category):
    q_str = ""
    for words in q.split():
        word = helper.removepunc(words)
        q_str = q_str + word + " "
    q_tokens  = nltk.word_tokenize(q_str)
    q_tag = nltk.pos_tag(q_tokens)
    ne_tag = nltk.ne_chunk(q_tag)
    for tree in ne_tag.subtrees():
        if tree.label() == category:
            return True
    return False

def contains(L,string):
    for q in L.split():
        if q == string:
            return True
    return False

def containsPOSTag(L,string):
    L_tag = PosTag(L)
    s = [True if x[1] == string else False for x in L_tag]
    return s

def containsList(L,lt):
    for q in L.split():
        if lt == constants.PERSON_NAMES or lt == constants.TIME:
            if q in lt:
                return True
        else:
            if q.lower() in lt:
                return True
    return False

def containsList_lemma(L,lt):
    q_lm = ps.stem(L.lower())
    for q in q_lm.split():
        if q in lt:
            return True
    return False



def containsPP(q,string):
    q_tag = PosTag(q)
    for i,w1 in enumerate(q.split()):
        if w1 == string:
            if q_tag[i+1][0] == "IN":
                return True
    return False

def containsNPwithPP(s):
    q_tag = PosTag(s)
    proper_noun = [x[0] for x in q_tag if x[1] == "NNP" or x[1] == "NNPS"]
    containsPP(s,proper_noun)

def whoRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if (not containsNER(q,"PERSON") or not containsList(q, constants.OCCUPATION) or not containsList(q, constants.PERSON_NAMES)) and (containsNER(s,"PERSON") or containsList(s,constants.OCCUPATION) or containsList(s,constants.PERSON_NAMES) or contains(s,"said")):
        score = score + constants.confident
    if (not containsNER(q,"PERSON") or not containsList(q, constants.OCCUPATION) or not containsList(q, constants.PERSON_NAMES)) and contains(s,"name"):
        score = score + constants.good_clue
    if containsPOSTag(s,"NNP") or containsPOSTag(s, "NNPS"):
        score= score + constants.good_clue
    return score

def whatRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if containsList(q, constants.MONTH) and containsList(s, ["today", "yesterday", "tomorrow", "last night"]):
        score = score + constants.clue
    if contains(q,"kind") and containsList_lemma(s, ["call", "from"]):
        score = score + constants.good_clue
    if contains(q,"name") or contains(q,"names") and containsList_lemma(s, ["name", "call", "known"]):
        score = score + constants.slam_dunk
    if containsPP(q, "name") and containsNPwithPP(s):
        score =  score + constants.slam_dunk
    return score

year_list = []
for i in range(1400,2000):
    year_list.append(str(i))
def whenRule(q,s):
    score = 0

    if containsList(s, constants.TIME) or containsList(s, year_list) :
        score = score + constants.good_clue
        score = score + WordMatch(q,s)
    if contains(q,"the last") and containsList(s, ["first", "last", "since", "ago"]):
        score = score + constants.slam_dunk
    if containsList(q,["start", "begin"]) and containsList(s, ["start", "begin", "since", "year"]):
        score = score + constants.slam_dunk
    return score

def whereRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if containsList(s, constants.locPrep):
        score = score + constants.good_clue
    if containsNER(s, "LOCATION") or containsList(s, constants.LOCATION) or containsNER(s, "GPE"):
        score = score + constants.confident
    return score

def whyRule(q,s, best, prev_sent, next_sent):
    score = 0
    for b_sent in best:
        if s == b_sent:
            score = score + constants.good_clue
        if next_sent == b_sent:
            score = score + constants.clue
        if prev_sent == b_sent:
            score = score + constants.clue
    if contains(s, "want"):
        score = score + constants.good_clue
    if containsList(s, ["so", "because"]):
        score = score + constants.good_clue
    return score

def datelineRule(q):
    score = 0
    if contains(q, "happen"):
        score = score + constants.good_clue
    if contains(q, "take") and contains(q, "place"):
        score = score + constants.good_clue
    if contains(q, "this"):
        score = score + constants.slam_dunk
    if contains(q, "story"):
        score = score + constants.slam_dunk
    return score

def whyMainRule(q,sent_list):
    score = []
    best = []
    temp_list = []
    for k,s in enumerate(sent_list):
        score.append(WordMatch(q,helper.remove_puncts(s)))
        best.append((s, score[k]))  # best is all the sentences with their scores
    best_list = sorted(best, key=lambda  x: (-x[1],x[0]))
    sent_best = []
    for j in range(10):
        sent_best.append(best_list[j][0])
    for i,s in enumerate(sent_list):
        if i == 0:
            score = whyRule(q,helper.remove_puncts(s),sent_best,None,sent_list[i+1])
        elif  i == len(sent_list) - 1:
            score = whyRule(q,helper.remove_puncts(s),sent_best,sent_list[i-1],None)
        else:
            score = whyRule(q,helper.remove_puncts(s),sent_best,sent_list[i-1],sent_list[i+1])
        temp_list.append((s,score))
    return temp_list

def howRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if containsList(s, constants.Numeric) or containsPOSTag(s,"CD"):
        score = score + constants.good_clue
    return score