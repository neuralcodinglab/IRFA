import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import IRFA as gen
import discriminator as dis
import os
from tqdm import tqdm
import torch.nn as nn
import torchvision
from typing import Tuple
import torch.nn.functional as F


def roi_to_electrodes(roi_train_on = ['V1', 'V4', 'IT' ]):
    
    '''
    
    Must be a list, for the monkey data, you can choose between V1, V4 or IT
    
    '''

    electrodes_v1         = [*(range(1,6))] + [*(range(7,9))] # electrode #6 broken
    electrodes_v4         = [*(range(9,13))]
    electrodes_IT         = [*(range(13,17))]

    roi_dic               = {'V1':electrodes_v1, 'V4':electrodes_v4, 'IT': electrodes_IT }

    roi_electrodes            = []

    for r in roi_train_on:
        roi_electrodes.extend(roi_dic[r])
    
    return roi_electrodes

class ROIReliabDataset(Dataset):
    def __init__(self, electrodes, set_t, grid_size):
        data_path = f'roi_reliab_all/{set_t}'

        self.targets = np.load(f'{data_path}/targets.npy').astype(np.float32)
        self.signals = []
        self.resizer = torchvision.transforms.Resize(size=(grid_size, grid_size))
        
        for e in electrodes:
            elec_paths = sorted(glob(f'{data_path}/electrodes/brain_signals_{set_t}_electrodes{str(e).zfill(2)}.npy'))            
            if elec_paths:
                for p in elec_paths:
                    signal_data = np.load(p).astype(np.float32)
                    self.signals.append(signal_data.transpose(1,0))
            else:
                print(f'No valid path for electrode {e}')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        signal_tensors = [torch.from_numpy(signal[idx, :]) for signal in self.signals]
        target_tensor = self.resizer(torch.from_numpy(self.targets[idx]).permute(2, 0, 1))
        return signal_tensors, target_tensor
    
if __name__ == '__main__':
    device = torch.device("cuda:4")
    
    grid_size = 64
    kernel_size = 4
    out_chan = 32 # Amount of features
    # -------------------------

    batch_size            = 8
    epochs          = 100
    load_ep = 0

    runname = f'IRF_multihead2_k{kernel_size}_o{out_chan}_l1vgg_l1_{lr}'

    save_dir = f'saved_models/{runname}/'
    # --- making paths ---
    os.makedirs(f'{save_dir}/train/losses', exist_ok = True)
    os.makedirs(f'{save_dir}/train/maps', exist_ok = True)
    os.makedirs(f'{save_dir}/current_arrays/', exist_ok = True)
    os.makedirs(f'{save_dir}/test/losses', exist_ok = True)
    os.makedirs(f'{save_dir}/test/maps', exist_ok = True)

    electrodes = roi_to_electrodes(['V1', 'V4', 'IT'])
    test_dataset = ROIReliabDataset(electrodes=electrodes, set_t='test', grid_size=grid_size)
    test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

    train_dataset = ROIReliabDataset(electrodes=electrodes, set_t='train', grid_size=grid_size)
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    
    generator = gen.Generator(h=15, 
                          c=out_chan, 
                          i=grid_size, 
                          k=kernel_size, 
                          alpha=0.01, 
                          beta_vgg=0.69, 
                          beta_pix=0.3,
                          lr=0.00078).to(device)
    
    discriminator = dis.Discriminator(out_chan).to(device)
    gen_lossfun   = gen.Lossfun(alpha_discr, beta_vgg, beta_pix).to(device)
    dis_lossfun   = dis.Lossfun(1).to(device)

    for ep in range(epochs):
        e = ep + load_ep

        for b, batch in enumerate(tqdm(train_set, total = len(train_set))):
            brains = [b.to(device) for b in batch[0]]  # Assuming batch returns a list of tensors

            targets = batch[-1].to(device)
            irfa_maps, M, P, _w = generator._irfa(brains)
            y_hat = generator.network(irfa_maps)
            
            if b % 100 == 0:
                dis_loss_train = discriminator.train(generator.network, irfa_maps, targets)
            
            generator.train(discriminator.network, brains, targets)

        if e%40 == 0:
            # Save the parameters of the Generator's network
            torch.save(generator.network.state_dict(), f'{save_dir}netG_{e}.pt')
            # Save the parameters of the Generator's attention extractor
            torch.save(generator._irfa.state_dict(), f'{save_dir}attention_extractor_{e}.pt')    
            # Save the parameters of the Discriminator's network
            torch.save(discriminator.network.state_dict(), f'{save_dir}netD_{e}.pt')

