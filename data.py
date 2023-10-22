import numpy as np
import torch
from torch_geometric.loader import DataLoader as GDataLoader

def get_data(args, isproxy, perm, dataset):
    indices = perm

    if isproxy == True:
        indices = np.array([i for i in range(len(dataset))])
        np.random.shuffle(indices)
        idx1, idx2 = int(len(dataset)*0.8), int(len(dataset)*0.9)
        # np.flip(indices)
    else:
        idx1 = args.num_train
        idx2 = idx1 + args.num_val
    
    train_mask, val_mask, test_mask = indices[:idx1], indices[idx1:idx2], indices[idx2:]

    masks = {"train": torch.tensor(train_mask).long().to(args.device),
             "val": torch.tensor(val_mask).long().to(args.device),
             "test": torch.tensor(test_mask).long().to(args.device)}

    selected_data = [dataset[i] for i in perm]

    train_data = selected_data[:idx1]
    val_data = selected_data[idx1:idx2]
    test_data = selected_data[idx2:]

    if isproxy == True:
        train_data = GDataLoader(train_data, shuffle=True, batch_size=32, num_workers=1)
        val_data = GDataLoader(val_data, shuffle=False, batch_size=32, num_workers=1)
        test_data = GDataLoader(test_data, shuffle=False, batch_size=32, num_workers=1)
        full_data = GDataLoader(dataset, shuffle=False, batch_size=6466, num_workers=1)
    else:
        train_data = GDataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=1)
        val_data = GDataLoader(val_data, shuffle=False, batch_size=args.batch_size, num_workers=1)
        test_data = GDataLoader(test_data, shuffle=False, batch_size=512, num_workers=1)
        full_data = GDataLoader(dataset, shuffle=False, batch_size=512, num_workers=1)

    data = {"train": train_data,
            "val": val_data,
            "test": test_data,
            "full": full_data
            }

    return masks, data


def ranking_data(args, ranking, perm, dataset):
    train_size, val_size = 100000, 10000

    index = (ranking == 1).nonzero()
    indices = index[torch.randint(0, index.shape[0], (train_size+val_size,))]
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data, val_data = [], []
    for index in train_indices:
        train_data.append((dataset[index[0]], dataset[index[1]]))
    for index in val_indices:
        val_data.append((dataset[index[0]], dataset[index[1]]))
    
    idx1 = -100
    idx2 = -50
    train_mask, val_mask, test_mask = perm[:idx1], perm[idx1:idx2], perm[idx2:]

    masks = {"train": torch.tensor(train_mask).long().to(args.device),
             "val": torch.tensor(val_mask).long().to(args.device),
             "test": torch.tensor(test_mask).long().to(args.device)}

    full_data = GDataLoader(dataset, shuffle=False, batch_size=6466)
    train_data = GDataLoader(train_data, shuffle=True, batch_size=8192)
    val_data = GDataLoader(val_data, shuffle=False, batch_size=8192)
    
    data = {"train": train_data,
            "val": val_data,
            "full": full_data
            }

    return masks, data