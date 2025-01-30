import argparse
import torch
from pathlib import Path
import h5py
import logging
from tqdm import tqdm
import pprint
import numpy as np

from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'NN': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'mutual_check': True,
            'distance_threshold': 0.7,
        },
    }
}

def get_model(conf):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    return model

@torch.no_grad()
def do_match (name0, name1, pairs, matched, num_matches_found, model, match_file, feature_file, query_feature_file, min_match_score, min_valid_ratio):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pair = names_to_pair(name0, name1)

    # Avoid to recompute duplicates to save time
    if len({(name0, name1), (name1, name0)} & matched) or pair in match_file:
        return num_matches_found
    data = {}
    feats0, feats1 = query_feature_file[name0], feature_file[name1]
    for k in feats1.keys():
        data[k+'0'] = feats0[k].__array__()
    for k in feats1.keys():
        data[k+'1'] = feats1[k].__array__()
    data = {k: torch.from_numpy(v)[None].float().to(device)
            for k, v in data.items()}

    # some matchers might expect an image but only use its size
    data['image0'] = torch.empty((1, 1,)+tuple(feats0['image_size'])[::-1])
    data['image1'] = torch.empty((1, 1,)+tuple(feats1['image_size'])[::-1])

    pred = model(data)
    matches = pred['matches0'][0].cpu().short().numpy()
    scores = pred['matching_scores0'][0].cpu().half().numpy()
    # if score < min_match_score, set match to invalid
    matches[ scores < min_match_score ] = -1
    num_valid = np.count_nonzero(matches > -1)
    if float(num_valid)/len(matches) > min_valid_ratio:
        v = pairs.get(name0)
        if v is None:
            v = set(())
        v.add(name1)
        pairs[name0] = v
        grp = match_file.create_group(pair)
        grp.create_dataset('matches0', data=matches)
        grp.create_dataset('matching_scores0', data=scores)
        matched |= {(name0, name1), (name1, name0)}
        num_matches_found += 1

    return num_matches_found


@torch.no_grad()
def best_match(conf, global_feature_path, feature_path, match_output_path, query_global_feature_path=None, query_feature_path=None, num_match_required=10,
               max_try=None, min_matched=None, pair_file_path=None, num_seq=False, sample_list=None, sample_list_path=None, min_match_score=0.85, min_valid_ratio=0.09):
    logging.info('Dyn Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')


    assert global_feature_path.exists(), feature_path
    global_feature_file = h5py.File(str(global_feature_path), 'r')
    if query_global_feature_path is not None:
        logging.info(f'(Using query_global_feature_path:{query_global_feature_path}')
        query_global_feature_file = h5py.File(str(query_global_feature_path), 'r')
    else:
        query_global_feature_file = global_feature_file

    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')
    if query_feature_path is not None:
        logging.info(f'(Using query_feature_path:{query_feature_path}')
        query_feature_file = h5py.File(str(query_feature_path), 'r')
    else:
        query_feature_file = feature_file

    match_file = h5py.File(str(match_output_path), 'a')

    if sample_list_path is not None:
        sample_list = json.load(open(str(sample_list_path, 'r')))

    # get all sample names
    if sample_list is not None:
        names = sample_list
        q_names = names
    else:
        names = []
        global_feature_file.visititems(
            lambda _, obj: names.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        names = list(set(names))
        names.sort()
        q_names = []
        query_global_feature_file.visititems(
            lambda _, obj: q_names.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        q_names = list(set(q_names))
        q_names.sort()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tensor_from_names(names, hfile):
        desc = [hfile[i]['global_descriptor'].__array__() for i in names]
        desc = torch.from_numpy(np.stack(desc, 0)).to(device).float()
        return desc

    desc = tensor_from_names(names, global_feature_file)
    if query_global_feature_path is not None:
        q_desc = tensor_from_names(q_names, query_global_feature_file)
    else:
        q_desc = desc
    # descriptors are normalized, dot product indicates how close they are
    sim = torch.einsum('id,jd->ij', q_desc, desc)
    if max_try is None:
        max_try = len(names)
    topk = torch.topk(sim, max_try, dim=1).indices.cpu().numpy()

    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    pairs = {}
    matched = set()
    for name0, indices in tqdm(zip(q_names, topk)):
        num_matches_found = 0
        # try sequential neighbor first
        if num_seq is not None:
            name0_at = names.index(name0)
            begin_from = name0_at - num_seq
            if begin_from < 0:
                begin_from = 0
            for i in range(begin_from, name0_at+num_seq):
                if i >= len(names):
                    break
                name1 = names[i]
                if name0 != name1:
                    num_matches_found = do_match(name0, name1, pairs, matched, num_matches_found, model, match_file, feature_file, query_feature_file, min_match_score, min_valid_ratio)

        # then the global retrievel
        for i in indices:
            name1 = names[i]
            if query_global_feature_path is not None or name0 != name1:
                num_matches_found = do_match(name0, name1, pairs, matched, num_matches_found, model, match_file, feature_file, query_feature_file, min_match_score, min_valid_ratio)
                if num_matches_found >= num_match_required:
                    break

        if num_matches_found < num_match_required:
            logging.warning(f'num match for {name0} found {num_matches_found} less than num_match_required:{num_match_required}')

    match_file.close()
    if pair_file_path is not None:
        if min_matched is not None:
            pairs = {k:v for k,v in pairs.items() if len(v) >= min_matched }
        pairs_list = []
        for n0 in pairs.keys():
            for n1 in pairs.get(n0):
                pairs_list.append((n0,n1))
        with open(str(pair_file_path), 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs_list))
    logging.info('Finished exporting matches.')

@torch.no_grad()
def main(conf, pairs, features, export_dir, db_features=None, query_features=None, output_dir=None, exhaustive=False):
    logging.info('Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    if db_features:
        feature_path = db_features
    else:
        feature_path = Path(export_dir, features+'.h5')
    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')

    if query_features is not None:
        logging.info(f'Using query_features {query_features}')
    else:
        logging.info('No query_features')
        query_features = feature_path
    assert query_features.exists(), query_features
    query_feature_file = h5py.File(str(query_features), 'r')

    pairs_name = pairs.stem
    if not exhaustive:
        assert pairs.exists(), pairs
        with open(pairs, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')
    elif exhaustive:
        logging.info(f'Writing exhaustive match pairs to {pairs}.')
        assert not pairs.exists(), pairs

        # get the list of images from the feature file
        images = []
        feature_file.visititems(
            lambda name, obj: images.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        images = list(set(images))

        pair_list = [' '.join((images[i], images[j]))
                     for i in range(len(images)) for j in range(i)]
        with open(str(pairs), 'w') as f:
            f.write('\n'.join(pair_list))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    match_name = f'{features}_{conf["output"]}_{pairs_name}'
    if output_dir is None:
        output_dir = export_dir
    match_path = Path(output_dir, match_name+'.h5')
    match_path.parent.mkdir(exist_ok=True, parents=True)
    match_file = h5py.File(str(match_path), 'a')

    matched = set()
    for pair in tqdm(pair_list, smoothing=.1):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        # Avoid to recompute duplicates to save time
        if len({(name0, name1), (name1, name0)} & matched) \
                or pair in match_file:
            continue

        data = {}
        feats0, feats1 = query_feature_file[name0], feature_file[name1]
        for k in feats1.keys():
            data[k+'0'] = feats0[k].__array__()
        for k in feats1.keys():
            data[k+'1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device)
                for k, v in data.items()}

        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((1, 1,)+tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,)+tuple(feats1['image_size'])[::-1])

        pred = model(data)
        grp = match_file.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)

        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)

        matched |= {(name0, name1), (name1, name0)}

    match_file.close()
    logging.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--output_dir', type=Path, required=False)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--db_features', type=Path)
    parser.add_argument('--query_features', type=Path, required=False)

    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--exhaustive', action='store_true')

    # best_match
    parser.add_argument('--best_match', action='store_true')
    parser.add_argument('--global_feature_path', type=Path)
    parser.add_argument('--feature_path', type=Path)
    parser.add_argument('--query_global_feature_path', type=Path)
    parser.add_argument('--query_feature_path', type=Path)
    parser.add_argument('--match_output_path', type=Path)
    parser.add_argument('--num_match_required', type=int, default=10)
    parser.add_argument('--min_matched', type=int, default=1)
    parser.add_argument('--max_try', type=int)
    parser.add_argument('--num_seq', type=int)
    parser.add_argument('--min_match_score', type=float, default=0.85)
    parser.add_argument('--min_valid_ratio', type=float, default=0.09)
    parser.add_argument('--sample_list_path', type=Path)
    parser.add_argument('--pair_file_path', type=Path)

    args = parser.parse_args()
    if args.best_match:
        best_match(confs[args.conf], args.global_feature_path, args.feature_path, args.match_output_path,
                   query_global_feature_path=args.query_global_feature_path, query_feature_path=args.query_feature_path,
                   num_match_required=args.num_match_required, min_matched=args.min_matched, min_match_score=args.min_match_score, min_valid_ratio=args.min_valid_ratio,
                   max_try=args.max_try, num_seq=args.num_seq, sample_list_path=args.sample_list_path, pair_file_path=args.pair_file_path)
    else:
        main(
            confs[args.conf], args.pairs, args.features,args.export_dir,
            db_features=args.db_features, query_features=args.query_features, output_dir=args.output_dir, exhaustive=args.exhaustive)
