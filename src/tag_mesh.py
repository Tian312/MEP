import os, json, sys, re
from pymetamap import MetaMap
mm = MetaMap.get_instance('/home/tk2624/tools/public_mm/bin/metamap20')
from postprocessing import attribute_processor
from general_utils.negex import negTagger
from general_utils import negex
rfile = open(r"general_utils/negation_triggers.txt")
irules = negex.sortRules(rfile.readlines())

attribute_processor = attribute_processor(mm,negTagger,irules)

all_files= os.listdir(sys.argv[1])
out_path = sys.argv[1]+"_new"
if not os.path.exists(out_path):
    try:
        os.mkdir(out_path)
    except:
        print("Error occurered when creating "+out_path+" .\n")

exception_list=open(os.path.join(out_path,"exception_list.txt"),"w")

count = 0

for f in all_files:
    if not re.search("json$",f):
        continue
    count +=1
    if count%50 ==0:
        print (count,"files processed...")
        
    jfile = open(os.path.join(sys.argv[1],f))
    outfile=os.path.join(out_path,f)
    if os.path.isfile(outfile):
        continue
    
    try:
        j_out = open(outfile,"w")
    
        data= json.load(jfile)
        for sent in data["Sentence-level breakdown"]:
            element= sent["Evidence Elements"]
            #new_elements_p=[]
            for p in element["Participant"]:
                term=p["term"]
                encode = attribute_processor.normalize(term)
                p["UMLS"]=encode
                #new_elements_p.append(p)
        
            for i in element["Intervention"]:
                term=i["term"]
                encode = attribute_processor.normalize(term)
                i["UMLS"]=encode

            for o in element["Outcome"]:
                term=o["term"]
                encode = attribute_processor.normalize(term)
                o["UMLS"]=encode

        json.dump(data, j_out)
        j_out.close()
    except:
        print (f,"error...")
        exception_list.write(f+"\n")
        
    

