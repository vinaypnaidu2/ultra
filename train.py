import os
import torch
import network
from torch import nn, optim
from torchvision.utils import save_image
import warnings
import numpy as np
from data_utils import train_loader, test_loader
from config import trainconfig
from metrics import psnr, ssim

warnings.filterwarnings('ignore')

def train(trainconfig):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    start_step = 0

    device = trainconfig.device
    model = network.B_transformer()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = trainconfig.lr, betas = (0.9, 0.999))
    mse = nn.L1Loss()

    if trainconfig.resume and os.path.exists(trainconfig.model_dir):
        print(f'Resume from {trainconfig.model_dir}')
        ckp = torch.load(trainconfig.model_dir, map_location = trainconfig.device)
        losses = ckp['losses']
        model.load_state_dict(ckp['model'])
        start_step = ckp['step']
        print(f'Resume training from step {start_step} :')
    else :
        print('Train from scratch :')

    for step in range(start_step, trainconfig.steps):
        model.train()
        batch = next(iter(train_loader))
        haze = batch[0].float().to(device)
        clear = batch[1].float().to(device)
        optimizer.zero_grad()            
        output = model(haze)   
        total_loss = mse(output, clear) 
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        print(f'\rTrain loss : {total_loss.item():.5f} | Step : {step}/{trainconfig.steps}', end = '', flush = True)
                        
        if not os.path.exists(trainconfig.save_dir):
            os.mkdir(trainconfig.save_dir)

        if step % trainconfig.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(model, test_loader)
            print(f'\nStep : {step} | SSIM : {ssim_eval:.4f} | PSNR : {psnr_eval:.4f}')
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)

            if ssim_eval > max_ssim and psnr_eval > max_psnr :
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                            'step' : step,
                            'max_psnr' : max_psnr,
                            'max_ssim' : max_ssim,
                            'ssims' : ssims,
                            'psnrs' : psnrs,
                            'losses' : losses,
                            'model' : model.state_dict()
                }, trainconfig.model_dir)
                print(f'\nModel saved at step : {step} | Best PSNR : {max_psnr:.4f} | Best SSIM : {max_ssim:.4f}\n')
            
            torch.save({
                        'step' : step,
                        'max_psnr' : max_psnr,
                        'max_ssim' : max_ssim,
                        'ssims' : ssims,
                        'psnrs' : psnrs,
                        'losses' : losses,
                        'model' : model.state_dict()
                }, trainconfig.model_dir[:-3] + '_fullepoch.pk')

            out_image = torch.cat([haze[0:3], output[0:3], clear[0:3]], dim = 0)
            save_image(out_image, trainconfig.save_dir + '/epoch{}.jpg'.format(step + 1))

def test(model, test_loader):
    model.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, (inputs,targets) in enumerate(test_loader):
        inputs = inputs.to(trainconfig.device); targets = targets.to(trainconfig.device)
        with torch.set_grad_enabled(False):
            pred = model(inputs)
            ssim1 = ssim(pred, targets).item()
            psnr1 = psnr(pred, targets)
            ssims.append(ssim1)
            psnrs.append(psnr1)
    return np.mean(ssims), np.mean(psnrs)

if __name__ == "__main__":
    train(trainconfig)
