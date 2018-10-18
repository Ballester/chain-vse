import os 
import types
import torch 
import torchvision 
import argparse
import numpy as np 
from glob import glob
from PIL import Image
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models


class CustomImageData(Dataset):

    def __init__(self, path, transforms=None):        
        self.path = path
        self.files = glob(os.path.join(path, '*.jpg'))
        self.n = len(self.files)
        print('Images found {}'.format(self.n))
        self.transforms = transforms

    def __len__(self, ):
        return self.n

    def __getitem__(self, idx):
        sel_file = self.files[idx]        
        file_name = sel_file.split('/')[-1]

        image = Image.open(sel_file).convert('RGB')         

        if self.transforms is not None:
            image = self.transforms(image)            
        
        return image, sel_file, file_name


def get_transform(nb_crops=1, crop_size=224, resize=256):

    # normalizer = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )
    
    resize = transforms.Resize(resize)
    if nb_crops == 1:
        crop = transforms.CenterCrop(crop_size)
        to_tensor = transforms.ToTensor()
    elif nb_crops == 10:
        crop = transforms.TenCrop(crop_size)                
        # normalizer = transforms.Lambda(
        #     lambda crops: torch.stack([normalizer(crop) for crop in crops], dim=0)
        # )
        to_tensor = transforms.Lambda(
            lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops], dim=0)
        )
    else:
        raise NotImplementedError('Number of crops not available.')
        
    transform = transforms.Compose(
        [resize, crop, to_tensor, ]
    )

    return transform


def modify_resnets(model):

    # Modify attributs
    model.last_linear = model.fc
    model.fc = None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods    
    model.forward = types.MethodType(features, model)
    return model


def get_network(arch='resnet152'):
    
    model = models.__dict__[arch](pretrained=True)
    model = modify_resnets(model)
    model = nn.DataParallel(model).cuda()

    return model


def get_rank(x):
    return len(x.size())


def extract(loader, model, outpath):

    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, total=len(loader))):
            
            images, fullpath, files = data
            images = images.cuda()
                            
            has_crops = (get_rank(images) == 5)            
            if has_crops: # 10crops
                b, c, f, h, w = images.size()
                images = images.view(b*c, f, h, w)
            
            images = torch.stack(
                [normalizer(x) for x in images], 0
            ).view(images.shape)
            
            preds = model(images)

            if has_crops: 
                pshape = preds.size() # batch*crops, features, height, width
                preds = preds.view(b, c, pshape[1], pshape[2], pshape[3])
                preds = preds.mean(1) # (batch, features, height, width)
            
            if i  == 0:
                tqdm.write('Feature shape {}'.format(preds.size()))
            
            export_preds(preds, files, outpath)


def export_preds(preds, names, outpath):
    
    for pred, name in zip(preds, names):        
        np.save(os.path.join(outpath, name), pred)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='/opt/datasets/chain/coco/images/',
                        help='path to datasets')
    parser.add_argument('--outpath', default=None,
                        help='path for saving the output features')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size [default=100]')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Preprocessing workers [default=4]')    
    parser.add_argument('--nb_crops', type=int, default=1,
                        help='Number of crops [default=1]')
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize to dim before crop [default=256]')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Crop size [default=256]') 
    
    args = parser.parse_args()    

    transform = get_transform(
        nb_crops=args.nb_crops, 
        crop_size=args.crop_size, 
        resize=args.resize
    )

    dataset = CustomImageData(
        path=args.datapath, 
        transforms=transform
    )    
    loader = DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers
    )
    
    model = get_network(arch='resnet152')

    extract(loader, model, args.outpath)
    

