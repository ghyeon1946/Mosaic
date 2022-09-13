import tensorflow as tf
import numpy as np
import pandas as pd
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from keras.models import load_model
import argparse

logger = logging.getLogger(__name__)

def device(self):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')
        
    return device

def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'NER_MODEL.h5'))

def load_model(pred_config, args, device):
    if not os.path.exists(self.args.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        self.model = load_model(self.args.model_dir, custom_objects = {"TFBertModel":TFBertModel})
        self.device(self.model)
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model

def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model

def predict(test_sentence):
  
  tokenized_sentence = np.array([tokenizer.encode(test_sentence, max_length=MAX_LEN, truncation=True, padding='max_length')])
  tokenized_mask = np.array([[int(x!=1) for x in tokenized_sentence[0].tolist()]])
  ans = model.predict([tokenized_sentence, tokenized_mask])
  ans = np.argmax(ans, axis=2)

  tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence[0])
  new_tokens, new_labels = [], []
  for token, label_idx in zip(tokens, ans[0]):
    if (token.startswith("##")):
      new_labels.append(dic_to_index[label_idx])
      new_tokens.append(token[2:])
    elif (token=='[CLS]'):
      pass
    elif (token=='[SEP]'):
      pass
    elif (token=='[PAD]'):
      pass
    elif (token != '[CLS]' or token != '[SEP]'):
      new_tokens.append(token)
      new_labels.append(dic_to_index[label_idx])

  for token, label in zip(new_tokens, new_labels):
      print("{}\t{}".format(label, token))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=10,
                help="epoch 를 통해서 학습 범위를 조절합니다.")
    parser.add_argument('--save_path', type=str, default='./checkpoint/',
                help="학습 결과를 저장하는 경로입니다.")
    parser.add_argument('--load_path', type=str, default='./checkpoint/Alls/KoGPT2_checkpoint_296000.tar', #
                help="학습된 결과를 불러오는 경로입니다.")
    parser.add_argument('--samples', type=str, default="samples/",
                help="생성 결과를 저장할 경로입니다.")
    parser.add_argument('--data_file_path', type=str, default='./short-story.csv',
                help="학습할 데이터를 불러오는 경로입니다.")
    parser.add_argument('--batch_size', type=int, default=8,
                help="batch_size 를 지정합니다.")

    pred_config = parser.parse_args()
    predict(pred_config)