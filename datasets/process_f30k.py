import argparse 
import json 
import os
import numpy as np
from glob import glob 
from tqdm import tqdm 
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_path', '-f')
    parser.add_argument('--ann_path', '-a')    
    parser.add_argument('--out_path', '-o')
    parser.add_argument('--normalize', action='store_true')
    return parser.parse_args()

args = get_args()

def main():

    f = glob('{}*.npy'.format(args.feat_path))
    print('Files found {}'.format(len(f)))
    if len(f) == 0:
        print('Error!')
        exit()

    with open('{}/dataset_flickr30k.json'.format(args.ann_path), 'r') as fp:
        dataset = json.load(fp)

    print('Anns loaded with {} keys'.format(len(dataset)))

    for split in ['train', 'val', 'test']:
        tqdm.write('Processing split {}'.format(split))
        inst = get_split_instances(dataset, split)
        feats, caps, ids = get_data(args.feat_path, inst)
    
        if split == 'train':
            tqdm.write('Fitting scaler')
            scaler = StandardScaler()
            scaler.fit(feats.mean(-1).mean(-1))
    
        # if args.normalize:
        #     tqdm.write('Applying standard scaler in {}'.format(split))            
        #     feats = scaler.transform(feats.mean(-1).mean(-1))

    
        tqdm.write('Saving {} set'.format(split))
        save(
            outpath=args.out_path, 
            feats=feats, 
            caps=caps, 
            ids=ids, 
            split=split,
            scaler=scaler,
        )


def save(outpath, feats, caps, ids, split, scaler=None):
    np.save(
        os.path.join(args.out_path, '{}_ims.npy'.format(split)), feats
    )
    np.save(
        os.path.join(args.out_path, '{}_ids.npy'.format(split)), ids
    )

    with open(os.path.join(outpath, '{}_caps.txt'.format(split)), 'w') as fp:
        for l in caps:
            fp.write('{}\n'.format(l))
            fp.flush()
    
    if scaler is not None:
        import cPickle as pickle
        with open(os.path.join(outpath, 'scaler.pkl'), 'w') as fp:
            pickle.dump(scaler, fp)


def get_split_instances(dataset, split):
    instances = defaultdict(list)
    for img in tqdm(dataset['images']):    
        id = img['filename']    
        if img['split'] == split:
            instances[id].extend([x['raw'] for x in img['sentences']])    
    return instances 


def get_data(feat_path, instances):
    features = []
    captions = []
    ids = []
    tqdm.write('Loading features')
    for k, v in tqdm(instances.items(), total=len(instances)):
        feat = np.load('{}/{}{}'.format(feat_path, k, '.npy'))
        feat = feat#.mean(-1).mean(-1)
        features.append(feat)
        ids.append(k)

        for _v in v:
            captions.append(_v)            
            
    features = np.asarray(features)
    tqdm.write('Features loaded: {}'.format(features.shape))
    return features, captions, ids

if __name__ == '__main__':
    
    main()