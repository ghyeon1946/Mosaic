
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging
import re
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from keras.models import load_model
import argparse

logger = logging.getLogger(__name__)

class Trainer(object):
   def __init__(self, args, train_dataset):
      self.args = args
      self.train_dataset = train_dataset
      self.index2dict
      self.dict2index
      self.train_x = []
      self.train_y = []
      self.targets = []
      self.target = []
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
      self.MAX_LEN = 90
      self.attention_masks = []
      self.model


   def device(self):
      if torch.cuda.is_available():    
         device = torch.device("cuda")
         print('There are %d GPU(s) available.' % torch.cuda.device_count())
         print('We will use the GPU:', torch.cuda.get_device_name(0))

      else:
         device = torch.device("cpu")
         print('No GPU available, using the CPU instead.')

   def DataSet(self):
      self.train_dataset = pd.read_csv(self.args.data_file_path, names=['src', 'cat'], sep="\t")
      self.train_dataset = self.train_dataset.reset_index()
      self.train_dataset[:30]

      sentence_x = []
      sentence_y = []

      for i in range(len(self.train_dataset)):
         if self.train_dataset['index'][i] != 1:
            sentence_x.append(self.train_dataset['src'][i])
            sentence_y.append(self.train_dataset['cat'][i])

         if self.train_dataset['index'][i] == 1:
            if len(sentence_x) != 0 and len(sentence_y) != 0:
               self.train_x.append(sentence_x)
               self.train_y.append(sentence_y)
            sentence_x = []
            sentence_y = []
            sentence_x.append(self.train_dataset['src'][i])
            sentence_y.append(self.train_dataset['cat'][i])

         print(self.train_x[:100])
         print(self.train_y[:100])

   def dict(self):
      self.index2dict = self.train_dataset['cat'].unique()
      self.index2dict = {word:i for i, word in enumerate(self.dict)}
      self.index2dict['[PAD]'] = 29

      self.dict2index = {i:word for i, word in enumerate(self.index2dict)}

   def Bert_Input(self):
      for i in range(len(self.train_x)):
         self.train_x[i]
         self.train_x[i].append('[SEP]')
         self.train_y[i].insert(0, '-')
         self.train_y[i].append('-')

         print(len(self.train_x))
         print(len(self.train_y))

         print(self.train_x[:100])
         print(self.train_y[:100])

         for i in range(len(self.train_y)):
            self.target = []
            for j in range(len(self.train_y[i])):
               self.target.append(self.index2dict[self.train_y[i][j]])
            self.targets.append(self.target)

         print(len(self.targets))

   def train(self):
      print(self.tokenizer.tokenize('나는 공주다'))

      tokenized_word = []
      tokenized_label = []
      tokenized_train_x = []
      tokenized_train_y = []

      print(len(self.targets))
      print(len(self.train_x))

      for i in range(len(self.train_x)):
         for j in range(len(self.train_x[i])):
            tokenized = self.tokenizer.tokenize(self.train_x[i][j])
            tokenized_label.extend([self.targets[i][j]] * len(tokenized))
            tokenized_word.extend(tokenized)
         tokenized_train_x.append(tokenized_word)
         tokenized_train_y.append(tokenized_label)
         tokenized_word = []
         tokenized_label = []

         print(tokenized_train_x[0])
         print(tokenized_train_y[0])

      input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_train_x]
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long", 
                                 value=self.tokenizer.convert_tokens_to_ids("[PAD]"), truncating="post", padding="post")
      
      input_ids_label = pad_sequences([i for i in tokenized_train_y], maxlen=self.MAX_LEN, 
                                       value=self.index2dict["[PAD]"], padding='post', dtype='int', truncating='post')

      for seq in input_ids:
         seq_mask = [float(i != self.tokenizer.convert_tokens_to_ids("[PAD]")) for i in seq]
         self.attention_masks.append(seq_mask)
      
      self.attention_masks = np.array(self.attention_masks)
      
      train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, input_ids_label, random_state=2018, test_size=0.1)

      train_masks, validation_masks, _, _ = train_test_split(self.attention_masks, input_ids, random_state=2018, test_size=0.1)
      
      train_inputs.tolist()
      validation_inputs.tolist()
      
      self.model = TFBertModel.from_pretrained("bert-base-multilingual-cased", from_pt=True, num_labels=len(self.index2dict), 
                                                output_attentions = False, output_hidden_states = False)
      
      token_inputs = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int64)
      mask_inputs = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int64)

      bert_outputs = self.model([token_inputs, mask_inputs])
      bert_outputs = bert_outputs[0]

      classifier = tf.keras.layers.Dense(len(self.index2dict), activation='softmax')(bert_outputs)

      self.model = tf.keras.Model([token_inputs, mask_inputs], classifier)

      optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
      self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00002), 
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['sparse_categorical_accuracy'])
            
      print(self.model.summary())
      
      self.model.fit([train_inputs, train_masks], train_labels, validation_data=([validation_inputs, validation_masks], validation_labels), epochs=3, batch_size=32)

   def save_model(self):
      if not os.path.exists(self.args.model_dir):
         os.makedirs(self.args.model_dir)

      self.model.save('NER_MODEL.h5')

      torch.save(self.args, os.path.join(self.args.model_dir, 'NER_MODEL.bin'))
      logger.info("Saving model checkpoint to %s", self.args.model_dir)

   def load_model(self):
      if not os.path.exists(self.args.model_dir):
         raise Exception("Model doesn't exists! Train first!")

      try:
         self.model = load_model(self.args.model_dir, custom_objects = {"TFBertModel":TFBertModel})
         self.device(self.model)
         logger.info("***** Model Loaded *****")
      except:
         raise Exception("Some model files might be missing...")