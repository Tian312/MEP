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
import os, re, sys, codecs
import collections, pickle
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')
from general_utils import formalization,tf_metrics,format_predict

from bert import modeling
from bert import optimization
from bert import tokenization
from parser_config import Config
config=Config()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
        "data_dir", None,
        "The input datadir.")

flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

from src.PICO_Recognition import PICO
from src.Medical_Evidence_Dependency import MED
PICO = PICO()
PICO_processor = PICO.get_processor()
PICO_estimator = PICO.get_estimator(PICO_processor)
MED = MED()
MED_processor = MED.get_processor()
MED_estimator = MED.get_estimator(MED_processor)

#bert tokenizer
tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=False)
# setting for quickUMLS
matcher = None
if config.use_UMLS > 0:
    from QuickUMLS.quickumls import QuickUMLS
    matcher = QuickUMLS(parser_config.QuickUMLS_dir,threshold=0.8)

def main():

    """ setting for output error list """
    if not os.path.exists(FLAGS.output_dir):
        try:
            createdir= "mkdir "+FLAGS.output_dir
            os.system(createdir)
        except:
            print("DIR ERROR! Unable to create this directory!")
    exception_dir = os.path.join(FLAGS.output_dir+"/exceptionlist.txt")
    except_out = codecs.open(exception_dir,"w")

    """ Processing each abstract in the folder """
    with tf.gfile.Open(os.path.join(config.bluebert_pico_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
        pico_label_list = PICO_processor.get_labels()
        print ("Start parsing PICO elements and dependency...")
        # start reading each file (abstract) and predict
        count = 1
        for f in os.listdir(FLAGS.data_dir):
            #pmid = re.sub("\.txt","",f)
            pmid = re.sub("\.\w+$","",f)
        
            #try:
            if 1 ==1:
                if not re.search("\.sents$", f) and not re.search("\.txt$",f):
                    continue
                input_file = os.path.join(FLAGS.data_dir, f)
                tags_file = input_file+".tags"
                #print (tags_file)
                abstract_text = codecs.open(input_file,"r").read()
                try:
                    tags = codecs.open(tags_file,"r").read().rstrip()
                    sent_tags= tags.split("\n")
                except:
                    sent_tags=[]
                #print (sent_tags)
                predict_examples = PICO_processor.get_pred_examples(input_file)
                predict_file = os.path.join(FLAGS.output_dir, "PICO.predict.tf_record")
                PICO_processor.filed_based_convert_examples_to_features(
                        predict_examples, 
                        pico_label_list,  
                        max_seq_length = 128, 
                        tokenizer = tokenizer, 
                        output_file = predict_file,
                        output_dir = FLAGS.output_dir,
                        mode = "test")
                predict_input_fn = PICO_processor.file_based_input_fn_builder(
                        input_file=predict_file,
                        seq_length=FLAGS.max_seq_length,
                        is_training=False,
                        drop_remainder=False)
                PICO_result = list(PICO_estimator.predict(input_fn=predict_input_fn))
                sents, sent_preds = PICO_processor.result_to_pair_for_return(predict_examples, PICO_result, id2label)
                
                
                # Start Parsing Medical Evidence Dependency MED
                sent_id = 0
                label_list=MED_processor.get_labels()
                sent_dict={}
                sent_json={}
                # prepare results:
                # sent_dict[sent_id] = sent_text
                # term_dict[term_id] = entity
                # entity_class_dict[term_id] = class
                # relation_list = [relation_id, sent_w_tag, label]
        
                for words,tags in zip(sents, sent_preds):
                               
                    #sent_id += 1
                    #tags = [re.sub("[B|I]-Count","O",a) for a in tags]
                    tags = format_predict.check_IOB(tags)
                    sent_dict[sent_id] = " ".join(words)
                    print("\n****"+"sent:",sent_id," ".join(words))
                    predict_examples, relation_list, term_dict, entity_class_dict = MED_processor.get_examples_from_pico(words,tags,str(sent_id))
                    #if len(predict_examples)<1:
                    #    print ("No")
                    num_actual_predict_examples = len(predict_examples)
                    predict_file = os.path.join(FLAGS.output_dir, "MED.predict.tf_record")
                    MED_processor.file_based_convert_examples_to_features(predict_examples, label_list,128, tokenizer, predict_file)
                    predict_input_fn = MED_processor.file_based_input_fn_builder( #share
                            input_file=predict_file,
                            seq_length=128,
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
                    print (relation_list_postive)
                    print (term_dict)
                    print (entity_class_dict)

                    sent_json[sent_id]= formalization.generate_json_from_sent(sent_id," ".join(words), term_dict, entity_class_dict, relation_list=[], umls_matcher=None)
                    sent_id += 1

                ''' writing to json '''
                outfile_dir= codecs.open(os.path.join(FLAGS.output_dir,pmid+".json"),"w")
                json_out = formalization.aggregate(pmid,abstract_text,sent_json,sent_tags)
                outfile_dir.write(json_out)
                #if count%50 ==0:
                #    print ("processing the",count,"th abstracts...")
                #count +=1
            #except:
            #    except_out.write(f+"\n")
        print ("Saved all parsing results in "+ FLAGS.output_dir)

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    main()
