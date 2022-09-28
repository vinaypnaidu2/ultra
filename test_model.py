import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import network
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch 
import numpy as np
# from skimage.measure import compare_ssim
# from skimage.measure import compare_psnr
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.utils import save_image
import network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_model = network.B_transformer().to(device)
my_model.eval()
my_model.to(device)


# my_model.load_state_dict(torch.load("./dehaze.pth", map_location = device)) 

checkpoint = torch.load('./nhhaze_both.pk', map_location=torch.device('cpu'))
my_model.load_state_dict(checkpoint['model']) 
my_model.eval()

to_pil_image = transforms.ToPILImage()


tfs_full = transforms.Compose([
            #transforms.Resize(1080),
            transforms.ToTensor()
        ])

def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.jpg' or '.png' in name]
    name_list.sort()
    return name_list
   
list_s = load_simple_list('./test/test_imgs2')
print(list_s)

for image in list_s:
    image_in = Image.open(image).convert('RGB')
    full = tfs_full(image_in).unsqueeze(0).to(device)
    output = my_model(full)
    save_image(output[0], './test/clear/nhhazebothclear{}'.format(image.split('/')[-1]))


