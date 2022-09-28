import os 
import argparse
import warnings
import torch
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type = int, default = 20000)
parser.add_argument('--resume', type = bool, default = True)
parser.add_argument('--eval_step', type = int, default = 100)
parser.add_argument('--lr', default = 0.0005, type = float, help = 'learning rate')
parser.add_argument('--dataset', type = str, default = 'nhhaze')
parser.add_argument('--type', type = str, default = 'both')
parser.add_argument('--bs', type = int, default = 10, help = 'batch size')
parser.add_argument('--ondevice', type = bool, default = False, help = 'to test training on device')
parser.add_argument('--save_dir', type = str, default = 'result')

trainconfig = parser.parse_args()
trainconfig.device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = trainconfig.dataset + '_' + trainconfig.type 
trainconfig.model_dir = './' + model_name +'.pk'

