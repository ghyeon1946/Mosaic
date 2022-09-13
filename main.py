import tensorflow as tf
import numpy as np
import sys
import os
from train import Model
import argparse
from train import Trainer

def main(args):
    train_dataset = args.data_file_path

    trainer = Trainer(args, train_dataset)

if __name__ == '__main__':
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

    args = parser.parse_args()