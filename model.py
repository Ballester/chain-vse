import torch
import torch.nn as nn
import torch.nn.init
import sys
import numpy as np
# sys.path.append('pretrained-models.pytorch/')
# import pretrainedmodels
import torchvision.models as models

import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.clip_grad import clip_grad_norm
from collections import OrderedDict
from torch.nn import functional as F
from text_encoders import get_text_encoder
from layers import l2norm


def print_summary(network):
    print(network)
    total = 0
    print('--'*30)
    for k, v in network.named_parameters():
        p = np.product(v.size())
        total += p
        if k.endswith('bias'):
            continue
        print('{:30s}: {:,}'.format(k, p))
    print('--'*30)
    print('Total parameters: {:,}'.format(total))
    print('--'*30)


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        is_tensor = ('tensor' in data_name)
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm, is_tensor=is_tensor)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
        else:
            model = nn.DataParallel(model)

        if torch.cuda.is_available():
            model.cuda()


        return model

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class Flatten(nn.Module):

    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims
    
    def forward(self, x):

        for dim in self.dims:
            x = x.squeeze(dim)

        return x
    

class EncoderImagePrecomp(nn.Module):

    def __init__(
            self, img_dim, embed_size, 
            use_abs=False, no_imgnorm=False, is_tensor=False
        ):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.is_tensor = is_tensor
        print('is tensor', is_tensor)

        fc = nn.Linear(img_dim, embed_size).cuda()
        
        self.fc = fc
        if is_tensor:
            conv = nn.Sequential(*[
                nn.Conv2d(img_dim, img_dim/2, 1, bias=False),
                nn.BatchNorm2d(img_dim/2),
                nn.ReLU(inplace=True),
                nn.Conv2d(img_dim/2, img_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(img_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(img_dim, embed_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_size),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),                
                Flatten(dims=(-1, -1)),
            ]).cuda()
            self.fc = conv
        
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        pass       

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
                
        features = self.fc(images)
        if len(features.shape) == 4:
            features.squeeze(-1).squeeze(-1)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2, keepdim=True).squeeze(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False,):
        super(ContrastiveLoss, self).__init__()        
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim        

    def forward(self, im, s, gamma=1.):
        '''
            when gamma == 1: using only hard_contrastives
                 gamma == 0: using all contrastives
        '''
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query        
        h_cost_s = cost_s.max(1)[0]
        h_cost_im = cost_im.max(0)[0]
        
        all_contr = cost_s.sum() + cost_im.sum()
        hrd_contr = h_cost_s.sum() + h_cost_im.sum()

        loss = all_contr * (1 - gamma) + hrd_contr * gamma
        return loss


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (
            torch.sum(torch.sum((mat ** 2), 1, keepdim=True), 2, keepdim=True).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt, ema=False):
        # tutorials/09 - Image Captioning
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)

        # if 'text_encoder' not in opt:
        #     opt.text_encoder = 'gru'
        #     opt.kwargs = {}
        #     opt.test_measure = None

        self.txt_enc = get_text_encoder(opt.text_encoder, opt)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(
            margin=opt.margin,
            measure=opt.measure
        )

        self.attention = False
        if opt.text_encoder.startswith('attentive'):
            self.init_attention()

        params = list(self.txt_enc.parameters())
        if self.img_enc.fc:
            params += list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        if ema:
            for param in filter(lambda p: p.requires_grad, self.params):
                param.detach_()
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.params), lr=opt.learning_rate)

        self.Eiters = 0

    def get_params(self):                
        params = list(self.img_enc.fc.parameters())
        params += list(self.txt_enc.parameters())
        if self.opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        return params

    def init_attention(self):
        opt = self.opt
        self.attention = True
        hops = self.txt_enc.hops
        self.I = Variable(torch.zeros(opt.batch_size, hops, hops))
        for i in range(opt.batch_size):
            for j in range(hops):
                self.I.data[i][j][j] = 1
        if torch.cuda.is_available():
            self.I = self.I.cuda()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        gamma = 1.
        if 'gamma' in kwargs:
            gamma = kwargs['gamma']
        loss = self.criterion(img_emb, cap_emb, gamma=gamma)
        if self.attention:
            coef = self.opt.att_coef
            attention = self.txt_enc.attention_weights
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - self.I[:attention.size(0)])
            total_loss = loss + coef * extra_loss

            self.logger.update('TotalLoss', total_loss.item(), img_emb.size(0))
            self.logger.update('AttLoss', coef * extra_loss.data[0], img_emb.size(0))

        self.logger.update('ContrLoss', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Iter', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, args['gamma'])

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()    

    def run_emb(self, images, captions, lengths, ids=None, *args):
        """Running embeddings for mean-teacher
        """
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        return img_emb, cap_emb
