from model.data_utils import Dataset,get_processing_word, minibatches
from model.models import HANNModel
from model.config import Config
import argparse
import codecs,re,time
import os,sys,warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
import warnings
warnings.filterwarnings("ignore")


from nltk.tokenize import sent_tokenize, word_tokenize

parser = argparse.ArgumentParser()
config = Config(parser) 
# build model
model = HANNModel(config)
model.build()
model.restore_session("study_arms/model.weights")

def preprocess_ab(line):  
        sents = sent_tokenize(line)
        sents = [" ".join(word_tokenize(s)) for s in sents ]
        return sents


def tag_sents(sents,model):
    for sent in sents:
        if sent == "":
            continue
        sent=re.sub("^\s*OBJECTIVES\s*\:?\s*|BACKGROUND\s*\:?\s*|DESIGN\s*\:?\s*|RESULTS\s*\:?\s*|CONCLUSIONS\s*\:?\s*|RESEARCH DESIGN AND METHODS\s*\:?\s*|SETTING\s*\:?\s*|MEASUREMENTS\s*\:?\s*|PARTICIPANTS\s*\:?\s*|RATIONALE\s*\:?\s*|METHODSs*\:?\s*","",sent)


def tag_abstract(raw_dir,sent_dir,file_name,model):
    tags= {0:"OBJECTIVE",1:"RESULTS",2:"METHODS",3:"BACKGROUND",4:"CONCLUSIONS"}
    lines_dir = os.path.join(sent_dir,"rawtext_byline")
    ab_text = codecs.open(os.path.join(raw_dir,file_name),"r").read()
    lines_file = codecs.open(lines_dir,'w')
    sents = preprocess_ab(ab_text)
   
    for sent in sents:
        if sent == "":
            continue
        sent=re.sub("^\s*OBJECTIVES\s*\:?\s*|BACKGROUND\s*\:?\s*|DESIGN\s*\:?\s*|RESULTS\s*\:?\s*|CONCLUSIONS\s*\:?\s*|RESEARCH DESIGN AND METHODS\s*\:?\s*|SETTING\s*\:?\s*|MEASUREMENTS\s*\:?\s*|PARTICIPANTS\s*\:?\s*|RATIONALE\s*\:?\s*|METHODSs*\:?\s*","",sent)
        lines_file.write("CONCLUSIONS\t"+sent+"\n")
    lines_file.write("\n\n")
    doc_id = re.sub("\.txt","",file_name)
    doc_name = doc_id+".sents"
    label_name = doc_id+".sents.tags"
    sents_outfile = codecs.open(os.path.join(sent_dir,doc_name),"w")
    tags_outfile = codecs.open(os.path.join(sent_dir,label_name),"w")
    data = Dataset(lines_dir, config.processing_word, config.processing_tag)
    for words, labels in data:
        labels_pred, document_lengths = model.predict_batch([words])
        i = 0
        for sent,pred in zip(sents,labels_pred[0]):
            if i == 0:
                tag = "TITLE"
                i+=1
            else:
                tag = tags[pred]
            sents_outfile.write(sent+"\n")
            tags_outfile.write(tag+"\n")
    rmcommand = "rm "+lines_dir
    os.system(rmcommand)


def main():
    if not os.path.exists(config.sents_dir):
        try:
            createdir= "mkdir "+config.sents_dir
            os.system(createdir)
        except:
            print("DIR ERROR! Unable to create this directory!")
    for f in os.listdir(config.raw_dir):
        print (f)
        if not re.search("\.txt",f):
            continue
        tag_abstract(config.raw_dir,config.sents_dir, f,  model)

if __name__ == "__main__":
    main()

