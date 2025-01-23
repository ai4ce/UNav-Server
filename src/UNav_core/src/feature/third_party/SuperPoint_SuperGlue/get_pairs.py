import torch
import h5py
import numpy as np

hfile = h5py.File('/home/endeleze/Desktop/WeNev/Localization/pytorch-NetVlad/Out/GS010021/global_features.h5', 'r')
q_hfile = h5py.File('/home/endeleze/Desktop/WeNev/Localization/pytorch-NetVlad/Out/GS010021/query/global_features.h5', 'r')

names = []
hfile.visititems(
    lambda _, obj: names.append(obj.parent.name.strip('/'))
    if isinstance(obj, h5py.Dataset) else None)
names = list(set(names))
db_names = [n for n in names]
q_names = []
q_hfile.visititems(
    lambda _, obj: q_names.append(obj.parent.name.strip('/'))
    if isinstance(obj, h5py.Dataset) else None)
q_names = list(set(q_names))
query_names = [n for n in q_names]


def tensor_from_names(names, hfile):
    desc = [hfile[i]['global_descriptor'].__array__() for i in names]
    desc = torch.from_numpy(np.stack(desc, 0)).to('cuda').float()
    return desc


db_desc = tensor_from_names(db_names, hfile)
query_desc = tensor_from_names(query_names, q_hfile)
sim = torch.einsum('id,jd->ij', query_desc, db_desc)
topk = torch.topk(sim, 10, dim=1).indices.cpu().numpy()

pairs = []
for query, indices in zip(query_names, topk):
    for i in indices:
        if query != db_names[i] :
            pair = (query, db_names[i])
            pairs.append(pair)

with open('/home/endeleze/Desktop/WeNev/Localization/pytorch-NetVlad/Out/GS010021/pairs.txt', 'w') as f:
    f.write('\n'.join(' '.join([i, j]) for i, j in pairs))