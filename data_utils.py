import os
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as tfs
from config import trainconfig

class RESIDE(data.Dataset):
    '''Dataset class to load RESIDE, RESIDE-NH and NH-Haze datasets'''
    
    def __init__(self, path, train, format = '.png'):
        super(RESIDE, self).__init__()
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        dim = (512, 512)
        haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        haze = haze.resize(dim)
        clear = clear.resize(dim)
        haze = haze.convert("RGB")
        clear = clear.convert("RGB") 
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
        return haze, clear

    def __len__(self):
        return len(self.haze_imgs)

if trainconfig.ondevice:
    path = '/Users/vinaynaidu/Desktop/'
else:
    path = '/kaggle/input/'

if trainconfig.dataset == "residenh":
    if trainconfig.type == "indoor":
        train_loader = DataLoader(dataset = RESIDE(path + 'residenh/RESIDENH/ITS', train = True), 
                                    batch_size = trainconfig.bs, shuffle = True)
        test_loader = DataLoader(dataset = RESIDE(path + 'residenh/RESIDENH/RESIDENH-VAL/indoor', train = False),
                                    batch_size = 1, shuffle = False)
    elif trainconfig.type == "outdoor":
        train_loader = DataLoader(dataset = RESIDE(path + 'residenh/RESIDENH/OTS', train = True),
                                    batch_size = trainconfig.bs, shuffle = True)
        test_loader = DataLoader(dataset = RESIDE(path + 'residenh/RESIDENH/RESIDENH-VAL/outdoor', train = False), 
                                    batch_size = 1, shuffle = False)
elif trainconfig.dataset == "reside":
    if trainconfig.type == "indoor":
        train_loader = DataLoader(dataset = RESIDE(path + 'resideh/RESIDE/ITS', train = True), 
                                    batch_size = trainconfig.bs, shuffle = True)
        test_loader = DataLoader(dataset = RESIDE(path + 'resideh/RESIDE/SOTS/indoor', train = False),
                                    batch_size = 1, shuffle = False)
    elif trainconfig.type == "outdoor":
        train_loader = DataLoader(dataset = RESIDE(path + 'resideh/RESIDE/OTS', train = True, format = '.jpg'),
                                    batch_size = trainconfig.bs, shuffle = True)
        test_loader = DataLoader(dataset = RESIDE(path + 'resideh/RESIDE/SOTS/outdoor', train = False), 
                                    batch_size = 1, shuffle = False)
elif trainconfig.dataset == "nhhaze":
        train_loader = DataLoader(dataset = RESIDE(path + 'nhhaze/NH-HAZE/NH-HAZE', train = True), 
                                    batch_size = trainconfig.bs, shuffle = True)
        test_loader = DataLoader(dataset = RESIDE(path + 'nhhaze/NH-HAZE/NH-HAZE-VAL', train = False),
                                    batch_size = 1, shuffle = False)

if __name__ == '__main__':
    pass
        