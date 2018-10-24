import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['unlabeled'] = {
            'img': os.path.join(imgdir, 'unlabeled'),
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, tokenizer, 
                 transform=None, ids=None, adapt_set=False):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            tokenizer: tokenizer wrapper.
            transform: transformer for image.
        """
        self.root = root
        self.split = None 
        self.data_path = root

        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.tokenizer = tokenizer
        self.transform = transform
        self.adapt_set = adapt_set

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        tokenizer = self.tokenizer
        root, caption, img_id, path, image = self.get_raw_item(index)

        # Convert caption (string) to word ids.
        target = tokenizer.tokenize_text(caption)
        
        if self.transform is not None:
            image_orig = self.transform(image)

            if self.adapt_set:
                image_adapt = self.transform(image)
                return (image_orig, image_adapt), target, index, img_id
                
        return image_orig, target, index, img_id

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')

        return root, caption, img_id, path, image

    def __len__(self):
        return len(self.ids)


class UnlabeledCocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, transform=None,):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            tokenizer: tokenizer wrapper.
            transform: transformer for image.
        """        
        self.root = root
        self.files = os.listdir(root)
        self.n = len(self.files)
        self.transform = transform
        self.data_path = root 
        self.split = 'unlabeled'

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """

        path = self.files[index]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            image_std = self.transform(image)
            image_ema = self.transform(image)

        return image_std, image_ema, -1, path, index

    def __len__(self):
        return self.n


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, tokenizer, transform=None, adapt_set=False):
        self.root = root
        self.data_path = root
        self.tokenizer = tokenizer
        self.split = split
        self.transform = transform
        self.adapt_set = adapt_set
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        tokenizer = self.tokenizer
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        # Convert caption (string) to word ids.
        target = tokenizer.tokenize_text(caption)
        
        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image_orig = self.transform(image)
            if self.adapt_set:
                image_adapt = self.transform(image)
                return (image_orig, image_adapt), target, index, img_id
        
        return image_orig, target, index, img_id

    def __len__(self):
        return len(self.ids)


class UnlabeledPrecompDataset(data.Dataset):

    def __init__(self, data_path, sigma=0.1, ):        
        self.features = np.load(os.path.join(data_path, 'unlabeled_ims.npy'))
        self.n = len(self.features)
        print(self.features.shape)        
        print('Loaded {} unlabeled image features'.format(self.n))
        self.sigma = sigma   
        self.data_path = data_path 
        self.split = 'unlabeled'
    
    def __len__(self,):
        return self.n
    
    def __getitem__(self, idx):
        feat_tensor = torch.FloatTensor(self.features[idx]) 

        noise = add_noise(feat_tensor, self.sigma)
        noise_ema = add_noise(feat_tensor, self.sigma)

        feat_tensor = feat_tensor + noise
        feat_tensor_ema = feat_tensor + noise_ema        
        # feat_tensor_ema = feat_tensor
        return feat_tensor, feat_tensor_ema, -1, idx, 1
        

def add_noise(x, sigma):
    return x+x.clone().normal_(0., sigma)


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, 
                tokenizer, sigma=0.01, adapt_set=False):
        self.split = data_split
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.adapt_set = adapt_set
        loc = data_path + '/'
        self.sigma = sigma

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        print(self.images.shape)
        self.length = len(self.captions)

        # rkiros data has redundancy in images
        # we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5            
        else:
            self.im_div = 1
            print(self.images.shape[0]//5)

        # the development set for coco is large and so validation would be slow
        if data_split == 'val':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div   
        image = torch.FloatTensor(self.images[img_id])

        image = add_noise(image, self.sigma)        

        caption = self.captions[index]
        tokenizer = self.tokenizer

        # Convert caption (string) to word ids.
        target = tokenizer.tokenize_text(caption)
        
        if self.adapt_set:
            image_ema = add_noise(image, self.sigma)
            return (image, image_ema), target, index, img_id
        
        return image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Loading for adapt set that requires different crops
    images_ema = None    
    if len(images[0]) == 2:
        images, images_ema = zip(*images)
        images_ema = torch.stack(images_ema, 0)        

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = list(map(len, captions))

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    if images_ema is not None:
        return images, images_ema, targets, lengths, ids

    return images, targets, lengths, ids


def get_loader_single(
        data_name, split, root,
        json, tokenizer, transform,
        batch_size=100, shuffle=True,
        num_workers=2, ids=None,
        collate_fn=collate_fn,
        adapt_set=False,
    ):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    if 'coco' in data_name:
        # COCO custom dataset
        dataset = CocoDataset(
            root=root,
            json=json,
            tokenizer=tokenizer,
            transform=transform,
            ids=ids,
            adapt_set=adapt_set,
        )

    elif 'f8k' in data_name or 'f30k' in data_name:
        dataset = FlickrDataset(
            root=root,
            split=split,
            json=json,
            tokenizer=tokenizer,
            transform=transform,
            adapt_set=adapt_set,
        )

    # It crashes when using CPU-only and pin_memory
    pin_memory = False
    if torch.cuda.is_available():
        pin_memory = True
    
    # Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loader


def get_precomp_loader(
        data_path, data_split, tokenizer, 
        opt, batch_size=100, shuffle=True, 
        num_workers=2, collate_fn=collate_fn, 
        adapt_set=False
    ):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    
    dset = PrecompDataset(
        data_path=data_path, 
        data_split=data_split, 
        tokenizer=tokenizer, 
        adapt_set=adapt_set,
        sigma=opt.noise,
    )

    pin_memory = False
    if torch.cuda.is_available():
        pin_memory = True

    data_loader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    return data_loader


def get_transform(data_name, split_name, opt):

    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split_name == 'train':
        t_list = [
            transforms.Scale(256),
            transforms.RandomSizedCrop(opt.crop_size),
            transforms.RandomHorizontalFlip(),
        ]
    elif split_name in ['val', 'test']:
        t_list = [
            transforms.Scale(256),
            transforms.CenterCrop(opt.crop_size),
        ]

    t_end = [transforms.ToTensor(), normalizer,]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loader(
        data_name, tokenizer, crop_size, 
        batch_size, workers, opt, 
        split='train', adapt_set=False,
    ):
    
    dpath = os.path.join(opt.data_path, data_name)    

    if opt.data_name.endswith('_precomp'):

        if split in ['train', 'val', 'test']:
            loader = get_precomp_loader(
                data_path=dpath, 
                data_split=split, 
                tokenizer=tokenizer, 
                opt=opt,
                batch_size=batch_size, 
                shuffle=(split == 'train'), 
                num_workers=workers, 
                collate_fn=collate_fn,
                adapt_set=adapt_set,
            )
        elif split == 'adapt':
            adapt_dataset = UnlabeledPrecompDataset(
                data_path=dpath,
                sigma=opt.noise,                
            )
            loader = torch.utils.data.DataLoader(
                dataset=adapt_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )

    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        if split in ['train', 'val', 'test']:
            transform = get_transform(data_name, split, opt)
   
            loader = get_loader_single(
                data_name=data_name,
                split=split,
                root=roots[split]['img'],
                json=roots[split]['cap'],
                tokenizer=tokenizer,
                transform=transform,
                ids=ids[split],
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=workers,
                collate_fn=collate_fn,
                adapt_set=adapt_set,
            )

        elif split == 'adapt':
            adapt_dataset = UnlabeledCocoDataset(
                root=roots['unlabeled']['img'],
                transform=transform,                
            )

            loader = torch.utils.data.DataLoader(
                dataset=adapt_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
            )
    
    return loader
    


def get_test_loader(
        split_name, data_name, tokenizer, 
        crop_size, batch_size, workers, 
        opt, collate_fn_str='collate_fn'
    ):

    cfn = eval(collate_fn_str)
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(
            dpath, split_name, tokenizer, opt,
            batch_size, False, workers
        )
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(
            opt.data_name, split_name,
            roots[split_name]['img'],
            roots[split_name]['cap'],
            tokenizer, transform, ids=ids[split_name],
            batch_size=batch_size, shuffle=False,
            num_workers=workers,
            collate_fn=collate_fn,
        )
    return test_loader


def get_tokenizer(vocab_path, data_name):

    from tokenizers import WordTokenizer
    from tokenizers import CharacterTokenizer

    if vocab_path.startswith('char'):
        tokenizer = CharacterTokenizer()
        vocab_size = tokenizer.encoder.n_alphabet
    else:
        vocab_path = os.path.join(vocab_path, '%s_vocab.pkl' % data_name)
        tokenizer = WordTokenizer(vocab_path)
        vocab_size = tokenizer.vocab_size

    return tokenizer, vocab_size