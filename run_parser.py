"""
May, 2020
author = "Tian Kang, Columbia University"
email = "tk2624@cumc.columbia.edu"

1. Sentence classifiction: Title, Ojective, Background, Methods, Results, Conclusion
2. NER: extracting PICO elements from abstracts in clinical literature
3. MED: Medical Evidence Dependency parsing
4. MEP: formulate Medical Evidence Proposition based on PICO and MED
"""

import logging
import os, re, sys, codecs,json
import collections, pickle
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] ="3"
from general_utils import formalization,tf_metrics
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from bert import modeling
from bert import optimization
from bert import tokenization
from parser_config import Config
config=Config()

flags = tf.flags
FLAGS = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
flags.DEFINE_string(
        "data_dir", None,
        "The input datadir.")

flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
        "NER_only", False,
        "recognize PICO elements only or parsing medical evidence dependecy and formulate propositions too.")


# load parser
from src.PICO_Recognition import PICO
from src.Medical_Evidence_Dependency import MED
PICO = PICO()
PICO_processor = PICO.get_processor()
PICO_estimator = PICO.get_estimator(PICO_processor)
MED = MED()
MED_processor = MED.get_processor()
MED_estimator = MED.get_estimator(MED_processor)
#from src import Evidence_Proposition_clustering

# load attribute tagger
from general_utils import negex
rfile = open(config.negation_rules)
irules = negex.sortRules(rfile.readlines())
mm = None
if config.use_UMLS:
    from pymetamap import MetaMap
    mm = MetaMap.get_instance(config.metamap_dir)
else:
    mm = None
from src.postprocessing import attribute_processor
attribute_processor = attribute_processor(mm,negex.negTagger,irules)

#bert tokenizer
tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=False)


def txt2ntokens(text):
    tokens = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    if len(tokens) >= int(max_seq_length) - 1:
        tokens = tokens[0:(max_seq_length - 2)]
    ntokens = []
    ntokens.append("[CLS]")
    for i, token in enumerate(tokens):
        ntokens.append(token)
        ntokens.append("[SEP]")
    return ntokens


def process_tag_sequence(words, tags):
    """ INPUT: one sentence pico results """
    tags = formalization.check_IOB(tags)
    
    #w_t = [(words[i],x) for i, x in enumerate(tags) if x != "O"]
    entity_list=[]
    entity_class_list=[]
    entity_negation_list=[]
    entity_encoding_list=[]
    word_tagged=[]
    sent = " ".join(words)
    last_tag = "O"
    
    for i, (word, tag) in enumerate(zip(words, tags)):
        #print (word, tag) 
        if re.search("^B",tag) :   #word tag :mild__B-Participant__T29  # HFABP__I-Outcome__O
            word_tagged.append(word+"__"+tag+"__"+str(len(entity_list)))             
            if last_tag =="O":
                entity_words = []
                entity_class = re.sub("B-","",tag)
                entity_words.append(word)
                entity_class_list.append(entity_class)
            else:
                term = " ".join(entity_words)
                entity_list.append(term)
            
                # negation, endocding for each recognized elements
                negation_tag = attribute_processor.detect_negation(term,sent)
                #encoding = attribute_processor.normalize(term)
                if config.use_UMLS:
                    encoding = attribute_processor.normalize(term)
                else:
                    encoding = {}
                entity_negation_list.append(negation_tag)
                entity_encoding_list.append(encoding)

                # first output last entity, then create new entity
                entity_words = []
                entity_class = re.sub("B-","",tag)
                entity_words.append(word)
                entity_class_list.append(entity_class)
                
            if i == len(tags)-1:
                term = " ".join(entity_words)
                entity_list.append(term)
            
                # negation, endocding for each recognized elements
                if entity_class == "Count":
                    encoding = {}
                    negation_tag = "affirmed"
                else:
                    negation_tag = attribute_processor.detect_negation(term,sent)
                    #encoding = attribute_processor.normalize(term)
                    if config.use_UMLS:
                        encoding = attribute_processor.normalize(term)
                    else:
                        encoding = {}
                entity_negation_list.append(negation_tag)
                entity_encoding_list.append(encoding)
                
        elif re.search("^I",tag):
            word_tagged.append(word+"__"+tag+"__O") 
            entity_words.append(word)
            if i == len(tags)-1:
                term = " ".join(entity_words)
                entity_list.append(term)
            
                # negation, endocding for each recognized elements
                
                if entity_class == "Count":
                    encoding = {}
                    negation_tag = "affirmed"
                else:
                    negation_tag = attribute_processor.detect_negation(term,sent)
                    #encoding = attribute_processor.normalize(term)
                    if config.use_UMLS:
                        encoding = attribute_processor.normalize(term)
                    else:
                        encoding = {}
                entity_negation_list.append(negation_tag)
                entity_encoding_list.append(encoding)
            
        else: # current tag = "O"
            word_tagged.append(word)
            if last_tag !="O":
                term = " ".join(entity_words)
                entity_list.append(term)
            
                # negation, endocding for each recognized elements
                
                if entity_class == "Count":
                    encoding = {}
                    negation_tag = "affirmed"
                else:
                    negation_tag = attribute_processor.detect_negation(term,sent)
                    #encoding = attribute_processor.normalize(term)
                    if config.use_UMLS:
                        encoding = attribute_processor.normalize(term)
                    else:
                        encoding = {}
                entity_negation_list.append(negation_tag)
                entity_encoding_list.append(encoding)
                
        last_tag = tag
    text_tagged = " ".join(word_tagged)
    #print ("\n"+" ".join(words))
    #print(text_tagged)
    #print (entity_list)
    #print (entity_class_list)

    assert len(entity_list) == len(entity_class_list)
    return entity_list, entity_class_list, entity_negation_list, entity_encoding_list,text_tagged


def main():

    """ setting for output error list """
    if not os.path.exists(FLAGS.output_dir):
        try:
            createdir= "mkdir "+FLAGS.output_dir
            os.system(createdir)
            print ("Creating output directory "+FLAGS.output_dir)
        except:
            print("DIR ERROR! Unable to create this directory!")
    exception_dir = os.path.join(FLAGS.output_dir+"/exceptionlist.txt")
    exception_list = []
    if os.path.isfile(exception_dir):
        exception_list = codecs.open(exception_dir,"r").read().rstrip().split("\n")
        except_out = codecs.open(exception_dir,"a")
        except_out = codecs.open(exception_dir,"w")
    else:
        except_out = codecs.open(exception_dir,"w")

    """ Processing each abstract in the folder """
    with tf.gfile.Open(os.path.join(config.bluebert_pico_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
        pico_label_list = PICO_processor.get_labels()
       
        # start reading each file (abstract) and predict
        count = 1
        print (" Prcoessing files from "+ FLAGS.data_dir)
        for f in os.listdir(FLAGS.data_dir):
            if f in exception_list:
                continue
            pmid = re.sub("\.\w+$","",f)
            if os.path.isfile(os.path.join(FLAGS.output_dir,pmid+".json")) and config.overwrite==False:
                print ("  <"+pmid+".json> exists. skipped")
                continue
            
            if True:
                #try:
                if not re.search("\.sents$", f) and not re.search("\.txt$",f):
                    continue
                input_file = os.path.join(FLAGS.data_dir, f)
                tags_file = input_file+".tags"
                abstract_text = codecs.open(input_file,"r").read()
                try:
                    tags = codecs.open(tags_file,"r").read().rstrip()
                    sent_tags= tags.split("\n")
                except:
                    sent_tags=[]
                
                """ Recognize Evidence Elements """
                predict_examples = PICO_processor.get_pred_examples(input_file)
                predict_file = os.path.join(FLAGS.output_dir, "PICO.predict.tf_record")
                PICO_processor.filed_based_convert_examples_to_features(
                        predict_examples, 
                        pico_label_list,  
                        max_seq_length = config.max_seq_length, 
                        tokenizer = tokenizer, 
                        output_file = predict_file,
                        output_dir = FLAGS.output_dir,
                        mode = "test")
                predict_input_fn = PICO_processor.file_based_input_fn_builder(
                        input_file=predict_file,
                        seq_length=config.max_seq_length,
                        is_training=False,
                        drop_remainder=False)
                PICO_result = list(PICO_estimator.predict(input_fn=predict_input_fn))
                sents, sent_preds = PICO_processor.result_to_pair_for_return(predict_examples, PICO_result, id2label, tokenizer)
                
                sent_id = 0
                label_list=MED_processor.get_labels()
                sent_dict={}
                sent_json={}
                
                
                """  Medical Evidence Dependency Parsing       
                     formulate Medical Evidence Propositions
                     if FLAGS.NER_only, output PICO elements results without parsing evidence dependency 
                """
                population = []
                MEP_list = [] # store MEP for cluster, to create the last level -- Medical Evidence Map

                for words,tags in zip(sents, sent_preds):

                    tags = formalization.check_IOB(tags)
                    sent_dict[sent_id] = " ".join(words)
                    entity_list, entity_class_list, entity_negation_list, entity_encoding_list,text_tagged = process_tag_sequence(words,tags)
                    
                    if FLAGS.NER_only:
                        sent_json[sent_id]= formalization.generate_json_from_sent(\
                                               sent_id," ".join(words),\
                                               entity_list=entity_list, \
                                               entity_class_list=entity_class_list,\
                                               entity_negation_list=entity_negation_list,\
                                               entity_encoding_list=entity_encoding_list,\
                                               NER_only=True)
                        sent_id += 1
                        continue

                    predict_examples, relation_list = MED_processor.get_examples_from_pico(entity_list, entity_class_list,text_tagged,str(sent_id))
                         
                    #print ("\n"+" ".join(words))
                    #print (entity_list)
                    #print (entity_class_list)
                    #print ("relation_list:",[a[0] for a in relation_list])

                    num_actual_predict_examples = len(predict_examples)
                    predict_file = os.path.join(FLAGS.output_dir, "MED.predict.tf_record")
                    MED_processor.file_based_convert_examples_to_features(predict_examples, label_list, config.max_seq_length, tokenizer, predict_file)
                    predict_input_fn = MED_processor.file_based_input_fn_builder( #share
                            input_file=predict_file,
                            seq_length=config.max_seq_length,
                            is_training=False,
                            drop_remainder=False)
                    MED_result = MED_estimator.predict(input_fn=predict_input_fn)
                    relation_list_postive=[]
                    for (i, prediction) in enumerate(MED_result):
                        p = prediction["probabilities"]
                        pred = "1" if p[0] < p[1] else "0"
                        if pred == "1":

                            relation_list[i][-1] = pred
                            relation_list_postive.append(relation_list[i][0])
                    
                    #print ("relation_list_postive:",relation_list_postive)
                    sent_json[sent_id]= formalization.generate_json_from_sent(sent_id," ".join(words),\
                                        entity_list = entity_list, \
                                        entity_class_list = entity_class_list, \
                                        entity_negation_list = entity_negation_list, \
                                        entity_encoding_list = entity_encoding_list, \
                                        relation_list= relation_list_postive,
                                        NER_only = False)

                    if True:
                        MEP_list.extend(sent_json[sent_id]["Evidence Propositions"])
                        population.extend(sent_json[sent_id]["Evidence Elements"]["Participant"])
                    sent_id += 1

                #arms_MEP, comp_MEP = Evidence_Proposition_clustering.cluster_mep(MEP_list)    
                
                # writing to json 
                outfile_dir= codecs.open(os.path.join(FLAGS.output_dir,pmid+".json"),"w")
                json_out = formalization.aggregate(pmid,abstract_text,sent_json,sent_tags)
                json_out["Evidence Map"]["Enrollment"] = population
                json_out["Evidence Map"]["Hypothesis"]=[]
                json_out["Evidence Map"]["Comparison Results"] = []#comp_MEP
                json_out["Evidence Map"]["Study Arm 1 Results"] = []#arms_MEP[0]
                json_out["Evidence Map"]["Study Arm 2 Results"] = []#arms_MEP[1]
                json_r=json.dumps(json_out)
                outfile_dir.write(json_r)
            
            #except:
            #    print("! error processing "+f+" . saved in exceptionlist.txt")
            #    except_out.write(f+"\n")       
            
            count += 1
            if count %50 == 0:
                print ("",count, "articles finished.")
        print ("Saved all parsing results in "+ FLAGS.output_dir)

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    main()
                       
    '''
    import time
    time0 = time.time()
    main()
    time1 = time.time()
    cost = (time1-time0) // 60
    print ("It cost:",cost,"min, or",time1-time0,"seconds.\n" )
    '''
    
