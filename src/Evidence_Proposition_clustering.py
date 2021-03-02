import os, sys, re
from bert_serving.client import BertClient  
bc=BertClient()
from sklearn.metrics.pairwise import cosine_similarity


def cluster_mep(mep_list):
    seed={0:[],1:[]}
    comp_mep=[]
    cand_mep=[]
    rules = ["both","either","2","two","each"]
    seed0="intervention"
    seed1="control"
    for mep in mep_list:
        if isinstance(mep["Intervention"],list) and len(mep["Intervention"])>1:
            seed0=mep["Intervention"][0]
            seed1=mep["Intervention"][1]
            comp_mep.append(mep)
        elif isinstance(mep["Intervention"],str):
            cand_mep.append(mep)
        else:
            comp_mep.append(mep)
    #print (seed0, seed1)
    for cand in cand_mep:
        intervention = cand["Intervention"]
        c = 0
        for r in rules: 
            if r in intervention.lower().split(" "):
                c = 1
                comp_mep.append(cand)
                continue
        if c == 1:
            continue
        emb = bc.encode([cand["Intervention"],seed0,seed1])
        sim = cosine_similarity(emb)[0]
        #print(sim, intervention)
        if sim[1]>sim[2]:
            seed[0].append(cand)
        else:
            seed[1].append(cand)

    return seed, comp_mep

        
mep_list = [{'Intervention': 'hydroxychloroquine', 'Observation': 'differ significantly', 'Outcome': 'incidence of new illness compatible with Covid-19', 'Count': ''}, {'Intervention': 'hydroxychloroquine', 'Observation': '', 'Count': '49 of 414 [ 11.8 % ]', 'Outcome': 'incidence of new illness compatible with Covid-19'}, {'Intervention': ['hydroxychloroquine', 'placebo'], 'Observation': 'more common', 'Outcome': 'Side effects', 'Count': ''},{'Intervention': 'hydroxychloroquine', 'Observation': '40.1 %', 'Outcome': 'Side effects', 'Count': ''},{'Intervention': 'placebo', 'Observation': '16.8 %', 'Outcome': 'Side effects', 'Count': ''}]
seed, comp_mep = cluster_mep(mep_list)
print (seed)
print(comp_mep)
