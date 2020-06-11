#/ usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT and NCBI-bluebert.

@Author:Tian Kang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import collections,codecs,os,sys,re,pickle
from nltk.tokenize import word_tokenize,sent_tokenize
#from general_utils import formalization,tf_metrics
from parser_config import Config
config=Config()

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')
#from tensorflow.python.util import deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

from bert import modeling
from bert import optimization
from bert import tokenization

flags = tf.flags
FLAGS = flags.FLAGS

max_seq_length= 128
do_train = False
do_eval = False
do_predict = True
use_tpu = False
learning_rate = 5e-5


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    @classmethod
    def raw2conll(self,raw_text):
        conll = []
        words = word_tokenize(raw_text)
        words =[[w,"O"] for w in words]
        conll.extend(words)
        conll.append([""])
        return conll


    def txt2conll(self, data_dir):
        conll = []
        raw_text = codecs.open(data_dir).read()
        if re.search("\.sents$",data_dir):
            sents = raw_text.split("\n")
        else:
            sents = sent_tokenize(raw_text)
        for sent in sents:
            if re.search("\w+\s+-\s+\w+",sent):
                sent = re.sub("\s+-\s+","-",sent)

            words = word_tokenize(sent) 

            words =[[w,"O"] for w in words]     
            conll.extend(words)
            conll.append([""])
        
        return conll

    def _read_data(cls, conll):
        """Reads a BIO data."""
        lines = []
        words = []
        labels = []

        for line in conll:

            if len(line) == 1:
                contends=[]
            else:
                contends = "\t".join(line)
   
            if len(contends) == 0:
                assert len(words) == len(labels)
                """ 
                if len(words) >30:
                    # split if the sentence is longer than 30
                    while len(words) > 30:
                        tmplabel = labels[:30]
                        for iidx in range(len(tmplabel)):
                            
                            if tmplabel.pop() == 'O':
                                break
                        l = ' '.join(
                            [label for label in labels[:len(tmplabel) + 1] if len(label) > 0])
                        w = ' '.join(
                            [word for word in words[:len(tmplabel) + 1] if len(word) > 0])

                        lines.append([l, w])
                        words = words[len(tmplabel) + 1:]
                        labels = labels[len(tmplabel) + 1:]
                """
                if len(words) == 0:
                    continue
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue

            word = line[0]
            label = line[-1]
            words.append(word)
            labels.append(label)
        
        return lines

class BC5CDRProcessor(DataProcessor):
    


    def get_train_examples(self, data_dir):
        l1 = self._read_data(os.path.join(data_dir,  "train.tsv"))
        l2 = self._read_data(os.path.join(data_dir, "devel.tsv"))
        return self._create_example(l1 + l2, "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")
    
    def get_pred_examples(self, data_dir,raw_text=False):
        if raw_text == False:
            return self._create_example(
            self._read_data(self.txt2conll(data_dir)), "test")
        else:
            return self._create_example(
            self._read_data(self.raw2conll(data_dir)), "test")
            
    
    def get_labels(self):
        #return ["O", "X","[CLS]", "[SEP]","B-Intervention","B-measure","B-modifier","B-Outcome","B-Participant","I-Intervention","I-measure","I-modifier","I-Outcome","I-Participant","Intervention","Outcome", "Participant"]
        #return ["O", "X","[CLS]", "[SEP]","B-Intervention","B-measure","B-modifier","B-Outcome","B-Participant","I-Intervention","I-measure","I-modifier","I-Outcome","I-Participant"]
        #return  ["O", "X","[CLS]", "[SEP]","B-Intervention", "B-Outcome","B-Participant","I-Intervention","I-Outcome","I-Participant","B-Count","I-Count","B-Observation","I-Observation"]
        return ["O", "X","[CLS]", "[SEP]","B-Intervention","B-Outcome","B-Participant","I-Intervention","I-Outcome","I-Participant","B-Observation","I-Observation","B-Count","I-Count"]
    
    def _create_example(self, lines, set_type):
        examples = []
        
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


    def filed_based_convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer, output_file, output_dir,mode=None):

        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
            feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode, output_dir)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            # features["label_mask"] = create_int_feature(feature.label_mask)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())


    def file_based_input_fn_builder(self, input_file, seq_length, is_training, drop_remainder):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            # "label_ids":tf.VarLenFeature(tf.int64),
            # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
        }

        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            return example

        def input_fn(params):
            batch_size = params["batch_size"]
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
            d = d.apply(tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder
            ))
            
            return d

        return input_fn

    def result_to_pair_for_return(self, predict_examples, predictions, id2label):
        """
        Args:
            predict_examples (list): InputExample, no X
            output: 
                words, list of words from input sentence (to be predicted)
                preds, list of pred tags for input sentence
        """

        words_all= []
        preds_all= []
        

        for predict_line, pred_ids in zip(predict_examples, predictions):
            words = str(predict_line.text).split(' ')
            labels = str(predict_line.label).split(' ')

            if len(words) != len(labels):
                tf.logging.error('Text and label not equal')

            if len(predict_examples) != len(predictions):
                tf.logging.error('{} vs {}'.format(len(predict_examples), len(predictions)))

            length = 0
            # get from CLS to SEP
            pred_labels = []
            for id in pred_ids:
                
                #print (curr_label)
                if id == 0:
                    continue
                curr_label = id2label[id] 
                if curr_label == '[CLS]':
                    continue
                elif curr_label == '[SEP]':
                    break
                elif curr_label == 'X':
                    continue
                pred_labels.append(curr_label)
            
            
            if len(pred_labels) > len(words):
                pred_labels = pred_labels[:len(words)]
            elif len(pred_labels) < len(words):
                pred_labels += ['O'] * (len(words) - len(pred_labels))
            
            words_all.append(words)
            preds_all.append(pred_labels)
            

        return words_all, preds_all

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode, output_dir):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    label2id_file = os.path.join(output_dir, 'label2id.pkl')
    if not tf.gfile.Exists(label2id_file):
        with tf.gfile.Open(label2id_file, 'wb') as w:
            pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= int(max_seq_length) - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # write_tokens(ntokens, label_ids, mode)
    return feature






def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
        #initializer=tf.contrib.layers.xavier_initializer() ##### new MODIFY ##########
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, max_seq_length, num_labels])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, predict)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                

                precision = tf_metrics.precision(label_ids, predictions, num_labels, [1, 2], average="macro")
                recall = tf_metrics.recall(label_ids, predictions, num_labels, [1, 2], average="macro")
                f = tf_metrics.f1(label_ids, predictions, num_labels, [1, 2], average="macro")
                #
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def result_to_pair(predict_examples, predictions, id2label, output_predict_file,output_err_file):
    """
    Args:
        predict_examples (list): InputExample, no X
    """
    if len(predict_examples) != len(predictions):
        tf.logging.error('{} vs {}'.format(len(predict_examples), len(predictions)))

    with tf.gfile.Open(output_predict_file, 'w') as writer, \
            tf.gfile.Open(output_err_file, 'w') as err_writer:
        for predict_line, pred_ids in zip(predict_examples, predictions):
            words = str(predict_line.text).split(' ')
            labels = str(predict_line.label).split(' ')
            
            
            
            if len(words) != len(labels):
                tf.logging.error('Text and label not equal')
                tf.logging.error(predict_line.text)
                tf.logging.error(predict_line.label)
                exit(1)
            # get from CLS to SEP
            pred_labels = []
            for id in pred_ids:
                if id == 0:
                    continue
                curr_label = id2label[id]             
                if curr_label == '[CLS]':
                    continue
                elif curr_label == '[SEP]':
                    break
                elif curr_label == 'X':
                    continue
                else:
                    pred_labels.append(curr_label)

       
            if len(pred_labels) > len(words):
                # tf.logging.error(predict_line.text)
                # tf.logging.error(predict_line.label)
                # tf.logging.error(words)
                # tf.logging.error(labels)
                # tf.logging.error(pred_labels)
                err_writer.write(predict_line.guid + '\n')
                err_writer.write(predict_line.text + '\n')
                err_writer.write(predict_line.label + '\n')
                err_writer.write(' '.join([str(i) for i in pred_ids]) + '\n')
                err_writer.write(' '.join([id2label.get(i, '**NULL**') for i in pred_ids]) + '\n\n')
                pred_labels = pred_labels[:len(words)]
            elif len(pred_labels) < len(words):
                # tf.logging.error(predict_line.text)
                # tf.logging.error(predict_line.label)
                # tf.logging.error(words)
                # tf.logging.error(labels)
                # tf.logging.error(pred_labels)
                err_writer.write(predict_line.guid + '\n')
                err_writer.write(predict_line.text + '\n')
                err_writer.write(predict_line.label + '\n')
                err_writer.write(' '.join([str(i) for i in pred_ids]) + '\n')
                err_writer.write(' '.join([id2label.get(i, '**NULL**') for i in pred_ids]) + '\n\n')
                pred_labels += ['O'] * (len(words) - len(pred_labels))

            for tok, label, pred_label in zip(words, labels, pred_labels):
                writer.write(tok + ' ' + label + ' ' + pred_label + '\n')
            writer.write('\n')



class PICO():
    def __init__(self):
        """
        PICO recognition model trained on BlueBERT
        """
        print ("Loading PICO recognition model...")

 
    def get_processor(self):
        processors = {
            "bc5cdr": BC5CDRProcessor,
        }

        task_name = "bc5cdr"#FLAGS.task_name.lower()  
        processor = processors[task_name]()
        return processor
    
    def get_estimator(self, processor):
        tokenizer = tokenization.FullTokenizer(
            vocab_file=config.vocab_file, do_lower_case=False)
        bert_config = modeling.BertConfig.from_json_file(config.bert_config_file) 
        label_list = processor.get_labels()
        tpu_cluster_resolver = None
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=config.bluebert_pico_dir,
            save_checkpoints_steps= 1000,#FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop= 1000,#FLAGS.iterations_per_loop,
                num_shards=8,#FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))
    
        num_train_steps = None
        num_warmup_steps = None

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list) + 1,
            init_checkpoint=config.init_checkpoint_pico,
            learning_rate=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu)

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            train_batch_size= 32 ,#FLAGS.train_batch_size,
            eval_batch_size= 8 ,#FLAGS.eval_batch_size,
            predict_batch_size= 8)
        return estimator