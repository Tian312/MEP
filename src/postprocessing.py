#/ usr/bin/env python3
# -*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import collections,codecs,os,sys,re,pickle
import os

semtypes_ex=["arch","bird","eico","crbs","euka","emst","enty","fish","fndg","geoa","gngm","grpa","grup","hcro","nusq","plnt","popg","qnco","qlco","rept","rnlw","shro","tmco","vtbt"]
        
class attribute_processor():
    def __init__(self,metamap,negTagger, negation_rules):
        self.metamap = metamap
        self.negTagger = negTagger
        self.negation_rules=negation_rules
        
    def detect_negation(self,term,sent):
        
        tagger = self.negTagger(sentence = sent, phrases =[term], rules = self.negation_rules, negP=False)
        tag=tagger.getNegationFlag()
        negation="affirmed"
        if tag=="negated":
            negation="negated"
        return negation

    def normalize(self,term):
        if self.metamap == None:
            return ""
        concepts,error = self.metamap.extract_concepts([term], restrict_to_sources=["MSH","SNOMEDCT_US","RXNORM","ATC","MTH"])#"MDR"
        encodings=[]
        #([ConceptMMI(index='USER', mm='MMI', score='5.18', preferred_name='Triplicate', cui='C0205174', semtypes='[qnco]', trigger='["Triplicate"-tx-1-"Triple"-verb-0]', location='TX', pos_info='1/6', tree_codes=''),
        pos_list=[]
        if len(concepts)>=0:
            for concept in concepts:
                if re.sub("[\[\]]","",concept[5]) in semtypes_ex:
                    continue
                if concept[-2] in pos_list:
                    continue
                try:
                    encodings.append({"preferred_name":concept[3],'cui':concept[4],"semtype":concept[5],"term":concept[6].split("\"")[3]})
                    pos_list.append(concept[-2])
                except:
                    continue
        return encodings

'''
from pymetamap import MetaMap          
mm = MetaMap.get_instance('/home/tk2624/tools/public_mm/bin/metamap20') 
from general_utils.negex import negTagger 
from general_utils import negex
rfile = open(r"general_utils/negation_triggers.txt")
irules = negex.sortRules(rfile.readlines())
attribute_processor = attribute_processor(mm,negTagger,irules)
t = ["dextran-70","apple"]
t= "duration of viral shedding"
s = "Experimental studies have demonstrated that dextran-70 does not reduce the leukocyte–endothelium interaction , but clinical evidence is still lacking . there is apple."
#n = attribute_processor.detect_negation(t,s)
encoding = attribute_processor.normalize(t)
print (encoding)
'''
