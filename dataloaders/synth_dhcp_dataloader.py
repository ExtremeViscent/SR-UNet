from sklearn import datasets
import torch
import torch.distributed as dist
from datasets import SynthdHCPDataset
from sklearn.model_selection import KFold
import numpy as np
import os
def get_transform(**kwargs):
    return None

def get_dataloader(data_dir=None,batch_size:int=1,**kwargs):
    dataset = SynthdHCPDataset(data_dir,**kwargs)
    dataset_val = SynthdHCPDataset(data_dir,phase='val',**kwargs)
    kfold = KFold(n_splits=5, shuffle=True)
    loaders = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        np.save(f"./output/train_ids_{fold}.npy", train_ids)
        np.save(f"./output/test_ids_{fold}.npy", test_ids)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, sampler=test_subsampler)
        loaders.append((trainloader, testloader))
    val_sampler = torch.utils.data.SubsetRandomSampler(range(len(dataset_val)))
    valloader = torch.utils.data.DataLoader(
                    dataset_val,
                    batch_size=batch_size, 
                    shuffle=False,
                    sampler=val_sampler)
    return loaders, valloader