import torch
import numpy as np

from torch.utils.data import DataLoader, random_split

np.random.seed(666)


def my_collate_fn(batch):
    src = torch.cat([b['src'].unsqueeze(0) for b in batch])
    tgt = torch.cat([b['tgt'].unsqueeze(0) for b in batch])
    src_sent_labels = torch.cat([b['src_sent_labels'].unsqueeze(0) for b in batch])
    segs = torch.cat([b['segs'].unsqueeze(0) for b in batch])
    clss = torch.cat([b['clss'].unsqueeze(0) for b in batch])
    mask_src = torch.cat([b['mask_src'].unsqueeze(0) for b in batch])
    mask_tgt = torch.cat([b['mask_tgt'].unsqueeze(0) for b in batch])
    mask_cls = torch.cat([b['mask_cls'].unsqueeze(0) for b in batch])
    src_txt = [b['src_txt'] for b in batch]
    tgt_txt = [b['tgt_txt'] for b in batch]

    return {
        "src": src,
        "tgt": tgt,
        "src_sent_labels": src_sent_labels,
        "segs": segs,
        "clss": clss,
        "mask_src": mask_src,
        "mask_tgt": mask_tgt,
        "mask_cls": mask_cls,
        "src_txt": src_txt,
        "tgt_txt": tgt_txt,
    }


def get_FL_dataloader(dataset, num_clients, split_strategy="Uniform",
                      do_train=True, batch_size=64,
                      do_shuffle=True, num_workers=0,
                      ):
    if split_strategy == "Dirichlet":
        # https://arxiv.org/pdf/2102.02079.pdf
        # https://github.com/Mangata1/NIID-Bench/blob/5371adbff98156793a413c7658923673b4aef7d7/utils.py#L179
        # Quantity Skew
        beta = 0.5
        idxs = np.random.permutation(len(dataset))
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        if do_train:
            client_datasets = [torch.utils.data.Subset(dataset=dataset, indices=batch_idxs[i]) for i in
                               range(num_clients)]
            trainloaders = []
            for ds in client_datasets:
                trainloaders.append(
                    DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers,
                               collate_fn=my_collate_fn))
            return trainloaders, client_datasets
        else:
            testloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=my_collate_fn)
            return testloader


    elif split_strategy == "Uniform":
        # Split training set into serval partitions to simulate the individual dataset
        partition_size = len(dataset) // num_clients
        lengths = [partition_size] * num_clients

        remainder = len(dataset) - (partition_size * num_clients)
        lengths[-1] += remainder

        if do_train:
            client_datasets = random_split(dataset, lengths, torch.Generator().manual_seed(666))
            trainloaders = []
            for ds in client_datasets:
                trainloaders.append(
                    DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers,
                               collate_fn=my_collate_fn))
            return trainloaders, client_datasets
        else:
            testloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=my_collate_fn)
            return testloader
    else:
        pass


