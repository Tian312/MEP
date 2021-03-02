import os,sys, re

# example command: 
#python run_parser.py --data_dir=data/test/txt --output_dir=data/test/json

#--------------------- MODIFY -------------------------------------#
#  Please modifie the following parameters before parsing.

class Config():

    # Base BERT config
    max_seq_length=256 
    vocab_file="model/biobert_config/vocab.txt"
    bert_config_file= "model/biobert_config/bert_config.json"
    pred_batch_size = 16
    do_lower_case = False
    learning_rate = 0.001
    overwrite = False  # overwrite the output json file in the target output directory
    
    # PICO NER config
    init_checkpoint_pico = "model/pico_model/model.ckpt"
    bluebert_pico_dir = "model/pico_model"
    
    covid_init_checkpoint_pico = "model/pico_covid_model/model.ckpt"
    covid_pico_dir = "model/pico_covid_model"

    # Medical Evidence Dependency config
    init_checkpoint_dependency = "model/dependency_model/model.ckpt"
    bluebert_dependency_dir = "model/dependency_model"
    
    # Sentence Classification config
    init_checkpoint_sent = ""
    bluebert_sent_dir = ""

    
    # attribute config
    negation_rules = "general_utils/negation_triggers.txt"
    metamap_dir= "/home/tk2624/tools/public_mm/bin/metamap20"
    use_UMLS = 0 # 0 represents not using UMLS
