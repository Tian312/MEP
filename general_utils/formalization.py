# coding: utf-8

import os, re,  string
import sys,codecs
#from general_utils.umls_tagging import get_umls_tagging
import json
import spacy
nlp = spacy.load("en_core_sci_lg")#"en_core_web_sm")
from itertools import product

import warnings
warnings.filterwarnings("ignore")

def check_IOB(preds):

    s = "=".join(preds)
    """
    O
    I-Outcome
    I-Outcome
    
    pattern=re.compile("O=I-")/
    err = re.findall(pattern,s)
    for e in err:
        match = re.findall("O=",e[0])
        new_e = "I-Outcome="+"I-Outcome="*len(match)+"I-Outcome"
        s = re.sub(e[0],new_e,s)
    new_s = re.sub("^I","B",s)
    new_preds=new_s.split("=")
    """
    new_s = re.sub("O=I","O=B",s)
    new_preds= new_s.split("=")
    return (new_preds)

def formulate_MEP(entity_list,entity_class_list, relation_list, observation_ids):
    """ example
            Do heavier women benefit from a higher dose of leuprolide acetate for suppression of serum estradiol ?
            new: ['0.2.3', '0.2.4']
            ['0.T2.T3', '0.T2.T4']
            {'T1': 'heavier women', 'T2': 'benefit', 'T3': 'higher dose of leuprolide acetate', 'T4': 'suppression of serum estradiol'}
            {'T1': 'Participant', 'T2': 'Observation', 'T3': 'Intervention', 'T4': 'Outcome'}

            Ob: T2
            MEP: I-Ob-O T3-T2-T4
    """
    """example
            At week 12 , whereas estradiol levels were significantly greater in the heavier patients in each of the groups ( LA 3.75 mg , p = 0.044 ; LA 7.5 mg , p = 0.002 ) , there was no significant difference in estradiol between groups as a whole or within any of the weight quartiles .
            ['7.T1.T2', '7.T2.T4', '7.T2.T5', '7.T2.T6', '7.T7.T8', '7.T7.T9']
            {'T1': 'estradiol levels', 'T2': 'significantly greater', 'T3': 'heavier', 'T4': 'each of the groups', 'T5': 'LA 3.75 mg', 'T6': 'LA 7.5 mg', 'T7': 'significant difference', 'T8': 'estradiol', 'T9': 'between groups'}
            {'T1': 'Outcome', 'T2': 'Observation', 'T3': 'Participant', 'T4': 'Intervention', 'T5': 'Intervention', 'T6': 'Intervention', 'T7': 'Observation', 'T8': 'Outcome', 'T9': 'Intervention'}

            Ob: T2 (JJR) T7 (JJ)
            MEP:  T4/T5-T2-T1; T9-T7-T8
            for relation in relation_list:
    """ 
    #JJS (Superlative Adjective Forms)
    #JJR 
    #RBS  (Superlative Adverbs)
    #RBR  (Comparative Adverb Forms)
    MEP_list = [] #[[I,Ob,O],[I, Ob,O]]
    comp_tags = ["JJS","JJR","RBS","RBR"]
    comp_terms = ["as","difference","different","same","similar","differences","more"]
    relation_list = [r.split(".")[1:] for r in relation_list]
    #for i,(e,t) in enumerate(zip(entity_list,entity_class_list)):
    #    print (i, e, t)

    for ob_id in observation_ids:
        
        ob_term = entity_list[int(ob_id)]
        ob_tags = [t.tag_ for t in nlp(ob_term)]
        flag = 0
        for comp in comp_tags:
            if comp in ob_tags:
                flag = 1
        for t in ob_term.split(" "):
            if t.lower() in comp_terms:
                flag = 1       
        
        if not flag:
            # type 1: single arm relation - observation tag is not in ["JJS","JJR","RBS","RBR"]
            I = []
            O = []
            for relation in relation_list:
                print (relation_list)
                ob_id= str(ob_id)
                if str(ob_id) not in relation:
                    continue
                relation.remove(ob_id)
                other_id = relation[0]
                if entity_class_list[int(other_id)] == "Outcome":
                    O.append(entity_list[int(other_id)])
                elif entity_class_list[int(other_id)]== "Intervention":
                    I.append(entity_list[int(other_id)])
            output = list(product(I, O)) # [(I,O),(I,O)]
            for o in output:
                mep = list(o)
                mep.insert(1, entity_list[int(ob_id)])
                MEP_list.append(mep)
        
        else:       # type 2: comparison relation
            '''
            ['1.0.1', '1.0.2']        
            '''

            I = []
            O = []
            for relation in relation_list:
                ob_id= str(ob_id)
                if str(ob_id) not in relation:
                    continue
                relation.remove(ob_id)
                other_id = relation[0]

                if entity_class_list[int(other_id)] == "Outcome":
                    O.append(entity_list[int(other_id)])
                elif entity_class_list[int(other_id)]== "Intervention":
                    I.append(entity_list[int(other_id)])
            if len(O)==0:
                MEP_list.append([I,entity_list[int(ob_id)], ""])
            for Outcome in O:
                MEP_list.append([I,entity_list[int(ob_id)], Outcome])
       
    return MEP_list         



def generate_json_from_sent(sent_id,sent_text, entity_list, entity_class_list, entity_negation_list, entity_encoding_list=[],relation_list=[], NER_only=True, sent_tag="METHODS"): # formulate json object for one sentence
    results = {}
    results["text"]= sent_text   
    results["annotated_text"]={}
    results["Evidence Elements"]={"Participant":[],"Intervention":[],"Outcome":[],"Observation":[],"Count":[]}
    results["Evidence Propositions"]=[]
    observation_class=[]
    count_class=[]
    for idx, entity in enumerate(entity_list):
        pico_element = entity
        pico_class = entity_class_list[idx]
        umls_tag=entity_encoding_list[idx]
        neg = entity_negation_list[idx]
        if pico_class not in ["Outcome", "Intervention","Participant"]:
            if pico_class == "Observation":
                observation_class.append(idx)
            else:
                count_class.append(idx)
            #continue
        try:
            start_index = sent_text.index(pico_element)
            end_index = start_index+len(pico_element)
        except:
            start_index = "NA"
            end_index = "NA"

        results["Evidence Elements"][pico_class].append({"term":pico_element,"negation":neg, "UMLS":umls_tag,"start":start_index, "end":end_index})
    
    if NER_only or len(relation_list)<1:

        return results
    
    MEP_list_ob = formulate_MEP(entity_list,entity_class_list, relation_list, observation_class)
    MEP_list_count = formulate_MEP(entity_list,entity_class_list, relation_list, count_class)
    
    for MEP in MEP_list_ob:
        results["Evidence Propositions"].append({"Intervention":MEP[0],"Observation":MEP[1],"Outcome":MEP[2], "Count":""})
    for MEP in MEP_list_count:
        results["Evidence Propositions"].append({"Intervention":MEP[0],"Observation":"","Count":MEP[1],"Outcome":MEP[2]})
    return results


    """ METHODS: We examined the association between hydroxychloroquine use and intubation or death at a large medical center in New York City"""
def aggregate(doc_id,abstract_text,sent_json,sentence_tags=[]):
    results = {}
    results["doc_id"] = doc_id
    results["type of study"]= "Therapy"
    results["title"]= ""

    results["abstract"] = abstract_text
    results["Evidence Map"]={}
    results["Evidence Map"]["Enrollment"]=[]
    #results["Evidence Map"]["study design"]={"Participant":"","Intervention":"", "Outcome":"","Hypothesis":""}
    results["Evidence Map"]["Comparison Results"]=[]
    results["Evidence Map"]["Study Arm 1 Results"]=[]
    results["Evidence Map"]["Study Arm 2 Results"]=[]
    results["Sentence-level breakdown"]=[]
    
    for sent_id in sent_json.keys():
        if len(sentence_tags)<1:
            sent_tag="METHODS"
        else:
            if sent_id > len(sentence_tags)-1:
                continue
            sent_tag = sentence_tags[sent_id]
            
        sent_ann = sent_json[sent_id]
        results["Sentence-level breakdown"].append({"Section":sent_tag,"Text":sent_ann["text"],"Evidence Elements":sent_ann["Evidence Elements"],"Evidence Propositions":sent_ann["Evidence Propositions"]})

    
    #json_r=json.dumps(results)
    return results


