"""
May, 2020
author = "Tian Kang, Columbia University"
email = "tk2624@cumc.columbia.edu"

1. Sentence classifiction: Title, Ojective, Background, Methods, Results, Conclusion
2. Evidence Element: extracting PICO elements from abstracts in clinical literature
3. Evidence Proposition: Medical Evidence Dependency parsing +formulate Medical Evidence Proposition based on PICO and MED
4. Evidence Map: Merge Evidence Propositions by study arms into Study Design and Study Results sections
"""

"""
This script is modified from run_parser.py to generate edge score matrix for Medical Evidence-informed Self-Attention
"""

import logging
import os, re, sys, codecs
import collections, pickle
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')
from general_utils import formalization,tf_metrics,format_predict
import numpy as np
np.set_printoptions(threshold=np.inf)
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




def dependency_matrix(words, relation_list_postive, term_dict, entity_class_dict ):
    
    """
    Example input:
    1 ONCLUSIONS : In this observational study involving patients with Covid-19 who had been admitted to the hospital , hydroxychloroquine administration was not associated with either a greatly lowered or an increased risk of the composite end point of intubation or death .
    ['1.T3.T4', '1.T4.T5']
    {'T1': 'Covid-19', 'T2': 'admitted to the hospital', 'T3': 'hydroxychloroquine administration', 'T4': 'greatly lowered or an increased', 'T5': 'risk of the composite end point of intubation or death'}
    {'T1': 'Participant', 'T2': 'Participant', 'T3': 'Intervention', 'T4': 'Observation', 'T5': 'Outcome'}

    direction: Outcome -> Observation/Count -> Intervention
    root Intervention is dependent on itself
    """ 
    
    def _get_loc(term_index, words, term_dict):
        """
        loc: the index of the first word of the term in the sent words
        """
        next_index = "T"+str(int(re.sub("T","",term_index))+1)
        if next_index in term_dict.keys():
            next_loc = " ".join(words).rindex(" "+term_dict[next_index])
            phrase = " ".join(words)[:next_loc]
        else:
            phrase = " ".join(words)
        loc_in_sent = phrase.rindex(term_dict[term_index])
        if loc_in_sent >0:
            loc = len(re.split("\s+",phrase[:(loc_in_sent)]))-1
        else:
            loc = 0 

        return loc

    dim = len(words)
    edge_score_matrix = np.zeros((dim,dim)) # (term1, term2) 
    #edge_score_matrix= np.empty((dim, dim), dtype = np.dtype('U100'))

    hierarchy={"Intervention":3,"Observation":2,"Count":1,"Outcome":0}
    for pair in relation_list_postive:
        
        if hierarchy[entity_class_dict[pair.split(".")[1]]]>hierarchy[entity_class_dict[pair.split(".")[2]]]:
            index_1 = pair.split(".")[1]
            index_2 = pair.split(".")[2]
        else:
            index_1 = pair.split(".")[2]
            index_2 = pair.split(".")[1]

        loc_1 = _get_loc(index_1, words, term_dict)
        loc_2 = _get_loc(index_2, words, term_dict)
        loc_list_1 = list(range(loc_1,len(term_dict[index_1].split(" "))+loc_1))
        loc_list_2 = list(range(loc_2,len(term_dict[index_2].split(" "))+loc_2))

        """ term 1: parent term loc_list_1 e.g. intervention
            term 2: child term  loc_list_2 e.g. observation/count
            1. assign 1 to edge_score_matrix[loc_1, loc_2] = 1
            2. if term 1 is intervention: 
                edge_score_matrix[loc_1,loc_1] = 1
            
        """
        import itertools
        coord = itertools.product(loc_list_1, loc_list_2)
        for c in coord:
            print (c,words[c[0]],words[c[1]],words[c[0]]+"-"+words[c[1]] )
            edge_score_matrix[c[0],c[1]] = 1 #str(words[c[0]]+"-"+words[c[1]])
        if entity_class_dict[index_1] == "Intervention":
            coord_intervention = itertools.product(loc_list_1, loc_list_1)
            for c in coord_intervention:
                edge_score_matrix[c[0],c[1]] = 1 # str(words[c[0]]+"-"+words[c[1]])

    return edge_score_matrix


def get_matrix_from_one_sent(sent_text,PICO_processor,PICO_estimator,MED_processor,MED_estimator):
    temp_out= os.path.join(os.getcwd(),"temp_out")
    if not os.path.exists(temp_out):
        try:
            createdir= "mkdir "+temp_out
            os.system(createdir)
        except:
            print("DIR ERROR! Unable to create this directory!")
            raise

    #start parsing
    with tf.gfile.Open(os.path.join(config.bluebert_pico_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
        pico_label_list = PICO_processor.get_labels()
        predict_examples = PICO_processor.get_pred_examples(sent_text, raw_text = True)
        predict_file = os.path.join(temp_out, "PICO.predict.tf_record")
        PICO_processor.filed_based_convert_examples_to_features(
                predict_examples, 
                pico_label_list,  
                max_seq_length = 128, 
                tokenizer = tokenizer, 
                output_file = predict_file,
                output_dir = os.getcwd(),
                mode = "test")
        predict_input_fn = PICO_processor.file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=False)
        PICO_result = list(PICO_estimator.predict(input_fn=predict_input_fn))
        sents, sent_preds = PICO_processor.result_to_pair_for_return(predict_examples, PICO_result, id2label)
        label_list=MED_processor.get_labels()
        words= sents[0]
        tags = sent_preds[0]
        tags = format_predict.check_IOB(tags)
        predict_examples, relation_list, term_dict, entity_class_dict = MED_processor.get_examples_from_pico(words,tags,"sent_0")
        predict_file = os.path.join(temp_out, "MED.predict.tf_record")
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
        matrix = dependency_matrix(words, relation_list_postive, term_dict, entity_class_dict)
        return words, matrix

def main():
    #test
    sent_text = "CONCLUSIONS : In this observational study involving patients with Covid-19 who had been admitted to the hospital , hydroxychloroquine administration was not associated with either a greatly lowered or an increased risk of the composite end point of intubation or death ." 
    w,m = get_matrix_from_one_sent(sent_text,PICO_processor,PICO_estimator,MED_processor,MED_estimator)
    print (w)
    print (m)


if __name__ == "__main__":
    main()
