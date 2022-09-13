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

def get_device(pred_config):
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
    if not os.path.exists(args.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = load_model(args.model_dir, custom_objects = {"TFBertModel":TFBertModel})
        model.to(device)
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model

def Read_Data(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines

def DataSet(lines, pred_config, args, tokenizer, pad_token_label_id):
    MAX_LEN = 90
    cls_token = '[CLS]'
    sep_token = '[SEP]'

    all_input_ids = []
    all_attention_mask = []

    for words in lines:
        tokens = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
        tokens.insert(0, cls_token)
        tokens.append(sep_token)

        input_ids = ([tokenizer.convert_tokens_to_ids(x) for x in tokens])
        all_input_ids.append(input_ids)

    all_input_ids = pad_sequences(all_input_ids, maxlen=MAX_LEN, dtype="long", 
                                value=tokenizer.convert_tokens_to_ids("[PAD]"), truncating="post", padding="post")

    for seq in all_input_ids:
        seq_mask = [float(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in seq]
        all_attention_mask.append(seq_mask)

    return all_input_ids, all_attention_mask

def get_labels():
    labels = {0: 'PER_B',
        1: 'DAT_B',
        2: '-',
        3: 'ORG_B',
        4: 'CVL_B',
        5: 'NUM_B',
        6: 'LOC_B',
        7: 'EVT_B',
        8: 'TRM_B',
        9: 'TRM_I',
        10: 'EVT_I',
        11: 'PER_I',
        12: 'CVL_I',
        13: 'NUM_I',
        14: 'TIM_B',
        15: 'TIM_I',
        16: 'ORG_I',
        17: 'DAT_I',
        18: 'ANM_B',
        19: 'MAT_B',
        20: 'MAT_I',
        21: 'AFW_B',
        22: 'FLD_B',
        23: 'LOC_I',
        24: 'AFW_I',
        25: 'PLT_B',
        26: 'FLD_I',
        27: 'ANM_I',
        28: 'PLT_I',
        29: '[PAD]'}

    return labels


def predict(pred_config):
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    
    labels = get_labels(args)
    logger.info(args)

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    lines = Read_Data(pred_config)
    dataset_ids, dataset_mask = DataSet(lines, pred_config, args, tokenizer, pad_token_label_id)

    ans = model.predict([dataset_ids, dataset_mask])
    ans = np.argmax(ans, axis=2)

    tokens = tokenizer.convert_ids_to_tokens(dataset_ids[1])
    new_tokens, new_labels = [], []
    
    with open("/content/drive/MyDrive/대학 수업/4-1/창설/Naver-NER/sample_pred_out.txt", "w", encoding="utf-8") as f:
        for i in range(len(ans)):
            tokens = tokenizer.convert_ids_to_tokens(all_input_ids[i])
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, ans[i]):
                if (token.startswith("##")):
                    new_labels.append(labels[label_idx])
                    new_tokens.append(token[2:])
                elif (token=='[CLS]'):
                    pass
                elif (token=='[SEP]'):
                    pass
                elif (token=='[PAD]'):
                    pass
                elif (token != '[CLS]' or token != '[SEP]'):
                    new_tokens.append(token)
                    new_labels.append(labels[label_idx])

            for token, label in zip(new_tokens, new_labels):
                line = ''
                print("{}\t{}".format(label, token))
                if label == '-':
                    line = line + token
                else:
                    line = line + "[{}:{}] ".format(token, label)
                f.write("{}\n".format(line.strip()))

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