import sys
import os, glob
import string
import nltk
import re
import textwrap
import copy
import numpy
import constants
import Rules
import Keywordcount
import helper
import FinalizeSent
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.data.path.append("/home/sandeep/nltk_data")
from nltk.corpus import state_union
#java_path = "C:/Program Files/Java/jdk1.8.0_20/bin/java.exe"
#os.environ['JAVAHOME'] = java_path
#nltk.internals.config_java("C:/Program Files/Java/jdk1.8.0_60/bin/java.exe")
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordNERTagger
#st = StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')
#st = StanfordNERTagger("stanford-ner-2015-04-20/classifiers/english.muc.7class.distsim.crf.ser.gz", "stanford-ner-2015-04-20/stanford-ner.jar")
from nltk.stem import PorterStemmer
ps = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
STOPWORDS = [lemmatizer.lemmatize(t) for t in stopwords.words('english')]
quoted = re.compile('"([^"]*)"')
os.chdir(sys.argv[1])
o = open('../out.txt', 'w')
ans_file = open("ans_file.txt",'w')
countfile = open("countlist.txt", 'w')
topfile = open("topsents.txt", 'w')
finalsents = open("finalsents.txt", 'w')
files = sorted(glob.glob('*.questions'))
cnt = 0
for file in files:
    fileId = file.title().split('.')[0]
    quotations = []
    ner_tag = []
    nnp_words = []
    complNominal = []
    otherNominal = []
    nounAdj = []
    otherNoun = []
    verb = []
    adverb = []
    cnt = cnt + 1
    print(cnt)
    fileId = file.title().split('.')[0]
    ques_list = []
    qid_list = []
    with open(file, 'r') as g:
        ques_lines = g.readlines()
        for line in ques_lines:
            if line.__contains__("QuestionID:"):
                qid_list.append(line)
            if line.__contains__("Question:"):
                ques = line.split("Question:")[1]
                ques_list.append(ques)
                Wh_word = helper.findWH(ques)
                #print Wh_word
                q_tagged = helper.Postagger(ques)
                q_tagged_copy = copy.deepcopy(q_tagged)
                #print "Answer Type: "
                #findNominal(q_tagged)
                #ner_tag.append(st.tag(ques[1].split()))
                #print ner_tag
                quot = []
                for val in quoted.findall(ques[1]):
                    if val:
                        quot.append(val)
                quotations.append(quot)
                #print quotations
                nnp_words.append(helper.findNNP(q_tagged))
                complNominal.append(Keywordcount.findcomplNominal(q_tagged))
                otherNominal.append(Keywordcount.findotherNominal(q_tagged))
                nounAdj.append(Keywordcount.findnounAdj(q_tagged))
                otherNoun.append(Keywordcount.findallNoun(q_tagged))
                helper.removestopwords_punct(q_tagged)
                verb.append(Keywordcount.findallVerb(q_tagged))
                adverb.append(Keywordcount.findallAdverb(q_tagged))
    with open(fileId + '.story', 'r') as f:
        story_text = f.read()
        header = story_text.split("TEXT:")[0]
        date = None
        if header.__contains__("DATE: "):
            dateline = header.split("DATE:")[1]
            date = dateline.split("\n")[0]
        #print date
        full_story = story_text.split("TEXT:")[1]
        '''
        i =0
        sent_list1=[]
        sent_list2=[]
        for para in full_story.split("\n\n")
            i++
            sent_list1=nltk.sent_tokenzie(para);

            sent_list2=sent_list2+sent_list1

        '''
        sent_list1 = nltk.sent_tokenize(full_story)
        sent_list = []
        for sent1 in sent_list1:
            sent1 = sent1[:-1]
            sent_list.append(sent1)
        '''
        sent_list_with_para=[]
         para_list = re.split("[\.|\"|!]\s*\n\n+", full_story)
         for sentstring in sent_list:
            for g,parastring in enumerate(para_list):
                if parastring.__contains__(sentstring)
                sent_list_with_para.append((sentstring,g))
                break

        '''
        sent_list_copy = copy.deepcopy(sent_list)  #sent_list is a list of sentences but not tagged. Each element of list is sentence which is not a list
        #sent_list = helper.remove_puncts(sent_list_copy)
    for z,q in enumerate(ques_list):
        count_list = []
        question_type = ""
        q_li = q.split()
        for x in reversed(q_li):
            if x.lower() == "who":
                question_type="who"
                break
            if x.lower() == "what":
                if q.__contains__("name") or q.__contains__("names"):
                    question_type = "who"
                    break
                question_type="what"
                break
            if x.lower() == "when":
                question_type="when"
                break
            if x.lower() == "where":
                question_type="where"
                break
            if x.lower() == "why":
                question_type="why"
                break
            if x.lower() == "how":
                for e,q_word in enumerate(q_li):
                    if q_word.lower() == "how" and "how " + q_li[e+1] not in constants.ans_type:
                        question_type="how"
                        break
                    if q_word.lower() == "how" and "how " + q_li[e+1] in constants.ans_type:
                        question_type="hownum"
                        break
        list1 = []
        dateline_score = 0
        if question_type=="who":
            for s in sent_list:
                score = Rules.whoRule(q,helper.remove_puncts(s))
                list1.append((s,score))
        elif question_type=="what":
            for s in sent_list:
                score = Rules.whatRule(q,helper.remove_puncts(s))
                list1.append((s,score))
        elif question_type=="when":
            dateline_score = Rules.datelineRule(q)
            for s in sent_list:
                score = Rules.whenRule(q,helper.remove_puncts(s))
                list1.append((s,score))
        elif question_type=="where":
            dateline_score = Rules.datelineRule(q)
            for s in sent_list:
                score = Rules.whereRule(q,helper.remove_puncts(s))
                list1.append((s,score))
        elif question_type=="hownum":
            for s in sent_list:
                score = Rules.howRule(q,helper.remove_puncts(s))
                list1.append((s,score))
        elif question_type=="why":
            list1 = Rules.whyMainRule(q,sent_list)
        else:
            for s in sent_list:
                s_nopunct = helper.remove_puncts(s)
                score = Rules.WordMatch(q,s_nopunct)
                list1.append((s,score))
        max_s = max(list1, key=lambda x:x[1])
        s = ""
        for x in list1:
            if question_type == "why":
                if x[1] == max_s[1]:
                    s = x[0]
            else:
                if x[1] == max_s[1]:
                    s = x[0]
                    break
        if question_type == "when" or question_type == "where":
            if dateline_score > max_s[1]:
                s = date
        if max_s[1] == 0:
            if question_type == "when" or question_type == "where":
                s = date
            elif question_type == "why":
                s = sent_list[len(sent_list)-1]
            else:
                s = sent_list[0]
        # Tie rule
        # list1 contains the sentences with scores
        #s_li = convertSenttolist(s)
        #s_li_copy = copy.deepcopy(s_li)
        #for word in s_li_copy:
            #if checkStopWord(word):
                #s_li.remove(word)
        #new_str = ""
        #for word in s_li:
            #new_str = new_str + word + " "
        s1 = s.strip()
        s2 = s1.replace("\n"," ")
        new_ans = []
        Whword = helper.findWH(q)
        s2_list = s2.split()
        # Prcess the s1.
        # Put if loop of answertype so that it seasy to debug. Currently who and where.
        # Put ner on s2 for the answer type
        # print 5 words before and and after save the value on file.
        list1 = helper.removezeroscore(list1)
        sorted_ans_list = sorted(list1, key=lambda  x: (-x[1]))
        paras = re.split("[\.|\"|!]\s*\n\n+", full_story)
        topfile.write("\n")
        topfile.write("\nQuestion: " + q)
        top_sents_cnt = 0
        for answ in sorted_ans_list:
            if top_sents_cnt < 5:
                top_sents_cnt = top_sents_cnt + 1
                topfile.write("\n" + answ[0] + "   ")
                topfile.write(str(answ[1]) + "\n")
                topfile.write("\n")
        top_ans = []
        for para in paras:
            para = para + "."
            count = 0
            count = count + Keywordcount.countexp(para, quotations[z])
            count = count + Keywordcount.countexp_nnp(para, nnp_words[z])
            count = count + Keywordcount.countexp(para, complNominal[z])
            count = count + Keywordcount.countexp(para, otherNominal[z])
            count = count + Keywordcount.countexp_na(para, nounAdj[z])
            count = count + Keywordcount.countexp_noun(para, otherNoun[z])
            count = count + Keywordcount.countexp_verb(para, verb[z])
            count = count + Keywordcount.countexp(para, adverb[z])
            count_list.append(count)
            top_cnt = 0
            para_sents = nltk.sent_tokenize(para)
            para_copy = copy.deepcopy(para_sents)
            #print(para_copy)
            #para_nopuncts = helper.remove_puncts(para_copy)
            #para_str = helper.convertListtoSent(para_nopuncts)
            #para_str = para_str.replace("\n", " ")
            for answer in sorted_ans_list:
                if top_cnt < 3:
                    top_cnt = top_cnt + 1
                    if para.__contains__(answer[0]) and count > 0:
                        top_ans.append(answer)
        countfile.write("\nQuestion: " + q)
        countfile.write(str(count_list))
        finalsents.write("\n")
        finalsents.write("Question: " + q)
        sorted_top_ans = sorted(top_ans, key=lambda  x: (-x[1]))
        sorted_top_ans1 = copy.deepcopy(sorted_top_ans)
        if len(sorted_top_ans) > 2:
            if sorted_top_ans1[0][1] - sorted_top_ans1[1][1] > 10:
                sorted_top_ans = []
                sorted_top_ans.append(sorted_top_ans1[0])
        for answ in sorted_top_ans:
            finalsents.write("\n" + answ[0] + "   ")
            finalsents.write(str(answ[1]) + "\n")
            finalsents.write("\n")
        if not top_ans == []:
            if question_type == "who" or question_type == "where":
                new_ans = FinalizeSent.matchFinalAnsWhoWhere(q,sorted_top_ans)
            if question_type == "what":
                new_ans = FinalizeSent.matchFinalAnsWhat(q,sorted_top_ans)
            if question_type == "when" or question_type == "hownum":
                new_ans = FinalizeSent.matchFinalAnsWhenHow(q,sorted_top_ans)
            if question_type == "why":
                new_ans = FinalizeSent.matchFinalWhy(q,sorted_top_ans)
        else:
            new_ans = FinalizeSent.formatFinalSent(Whword, s2, q)
            #new_ans = FinalizeSent.trimFinalAns(matched_ans,s2.split(),1)
        value = ""
        if len(new_ans) > 0:
            value_str = ""
            for ans in new_ans:
                #ans = " ".join("".join([" " if ch in string.punctuation else ch for ch in ans]).split())
                ans = ans.strip()
                ans = helper.removepunc(ans)
                value_str = value_str + ans + " "
            value = ""
            value_str_list1 = value_str.split()
            value_str = ""
            value_str_list = sorted(set(value_str_list1),key=value_str_list1.index)
            for val in value_str_list:
                if not helper.remqueswords(q,val):
                    value_str = value_str + str(val) + " "
            if value_str == "":
                match_ans = sorted_top_ans[0][0]
                match_ans1 = match_ans.strip()
                match_ans2 = match_ans1.replace("\n"," ")
                for val in match_ans2.split():
                    if not helper.remqueswords(q,val):
                        val = helper.removepunc(val)
                        value_str  = value_str + str(val) + " "
            value = qid_list[z] + "Answer: " + value_str + "\n\n"
        else:
            #s2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in s2]).split())
            value_str = ""
            for val in s2.split():
                if not helper.remqueswords(q,val):
                    val = helper.removepunc(val)
                    value_str  = value_str + str(val) + " "
            if value_str == "":
                match_ans = sorted_top_ans[0][0]
                match_ans1 = match_ans.strip()
                match_ans2 = match_ans1.replace("\n"," ")
                for val in match_ans2.split():
                    if not helper.remqueswords(q,val):
                        val = helper.removepunc(val)
                        value_str  = value_str + str(val) + " "
            value = qid_list[z] + "Answer: " + value_str + "\n\n"
        #value = " ".join("".join([" " if ch in string.punctuation else ch for ch in value]).split())
        #value = value.translate(None, string.punctuation)
        ans_file.write("\n")
        ans_file.write("Question: " + q)
        ans_file.write(value)
        ans_file.write("\n")
        o.write(value)
ans_file.close()
topfile.close()
countfile.close()
o.close()
