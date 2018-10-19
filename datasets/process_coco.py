import argparse 
import numpy as np
from pycocotools import coco
from glob import glob
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import cPickle as pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_path', '-f')
    parser.add_argument('--ann_path', '-a')    
    parser.add_argument('--out_path', '-o')
    parser.add_argument('--normalize', action='store_true')
    return parser.parse_args()

args = get_args()

def main():

    f = glob('{}/*/*.npy'.format(args.feat_path))
    print('Files found {}'.format(len(f)))
    if len(f) == 0:
        print('Error!')
        exit()

    coco1 = coco.COCO(annotation_file=os.path.join(args.ann_path, 'captions_train2014.json'))
    coco2 = coco.COCO(annotation_file=os.path.join(args.ann_path, 'captions_val2014.json'))

    print('Anns loaded')

    for split in ['train', 'val', 'test']:
        tqdm.write('Processing split {}'.format(split))
        
        load_ids = lambda x: list(np.load(os.path.join(args.ann_path, x)))
        if split == 'train':
            ids = load_ids('coco_train_ids.npy')
            ids += load_ids('coco_restval_ids.npy')
            
        else:
            ids = load_ids('coco_{}_ids.npy'.format(split.replace('val', 'dev')))

        tqdm.write('Loaded {} ids for {} set'.format(len(ids), split))

        instance_paths = get_split_instances(
            ids, path=args.feat_path, coco1=coco1, coco2=coco2)

        tqdm.write('Fetch {} paths'.format(len(instance_paths)))
        feats, caps = get_data(instance_paths)
        print('Features loaded {}'.format(feats.shape))
        print('Sanity check', feats.min(), feats.max(), feats.mean())

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


def get_split_instances(ids, path, coco1, coco2):
    features_path = OrderedDict()
    #captions = []
    for id in tqdm(sorted(ids)):
        try:
            instance = coco1.anns[id]
            img = coco1.imgs[instance['image_id']]
        except KeyError:
            instance = coco2.anns[id]
            img = coco2.imgs[instance['image_id']]
        
        caption = instance['caption']
        #captions.append(caption)
        folder = '_'.join(img['file_name'].split('_')[1:2])
        img_name = os.path.join(img['file_name'])
        feat_path = os.path.join(path, folder, img_name + '.npy')
        try:
            features_path[feat_path].append(caption)
        except KeyError:
            features_path[feat_path] = []
            features_path[feat_path].append(caption)

    return features_path


def get_data(features_path,):
    features = []
    captions = []
    for fpath, caption in tqdm(features_path.items()):
        feature = np.load(fpath).mean(-1).mean(-1)
        features.append(feature)
        captions.extend(caption)

    features = np.asarray(features)
    return features, captions

if __name__ == '__main__':
    
    main()