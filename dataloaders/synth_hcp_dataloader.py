from sklearn import datasets
import torch
import torch.distributed as dist
from datasets import SynthHCPDataset, SynthHCPDatasetTIO
from sklearn.model_selection import KFold
import numpy as np
import os
import torchio as tio


def get_dataloader(data_dir=None,batch_size:int=1,output_dir="./output", n_splits = 5, tio = False,**kwargs):
    if tio:
        dataset = SynthHCPDatasetTIO(data_dir, **kwargs)
        dataset_val = SynthHCPDatasetTIO(data_dir, **kwargs)
    else:
        dataset = SynthHCPDataset(data_dir,**kwargs)
        dataset_val = SynthHCPDataset(data_dir,phase='val',**kwargs)
    kfold = KFold(n_splits=n_splits, shuffle=True)
    loaders = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, 'train_ids_{}.npy'.format(fold)), train_ids)
        np.save(os.path.join(output_dir, 'test_ids_{}.npy'.format(fold)), test_ids)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=batch_size,
                        sampler=train_subsampler,)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, 
                        sampler=test_subsampler)
        loaders.append((trainloader, testloader))
    val_sampler = torch.utils.data.SubsetRandomSampler(range(len(dataset_val)))
    valloader = torch.utils.data.DataLoader(
                    dataset_val,
                    batch_size=batch_size, 
                    shuffle=False,
                    sampler=val_sampler)
    return loaders, valloader