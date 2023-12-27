"""
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch
import time
import numpy as np
import pdb
import cv2
import wandb
from torch import autograd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import sys
from argparse import Namespace
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

seed = 0
from numpy import random
random.seed(seed)
import face_alignment

from libs.utilities.utils import *
from libs.utilities.image_utils import *
from libs.DECA.estimate_DECA import DECA_model
#from libs.models.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.models.mask_predictor import MaskPredictor
from libs.utilities.stylespace_utils import decoder
from libs.configs.config_models import stylegan2_ffhq_1024
from libs.criteria.losses import Losses
from libs.criteria import id_loss
from libs.criteria.lpips.lpips import LPIPS
from libs.utilities.utils_inference import generate_grid_image, calculate_evaluation_metrics
from libs.utilities.dataloader import CustomDataset_validation

#from new_G import Generator_000500 as StyleGAN2Generator

from libs.models.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.models.mask_predictor import MaskPredictor
from libs.utilities.utils import make_noise, generate_image, generate_new_stylespace, save_image, save_grid, get_files_frompath
from libs.utilities.stylespace_utils import decoder
from libs.configs.config_models import stylegan2_ffhq_1024
from libs.utilities.utils_inference import preprocess_image, invert_image
from libs.utilities.image_utils import image_to_tensor
from libs.models.inversion.psp import pSp
from torchvision import transforms, utils
from PIL import Image
import random

###hyperparams
train_batch_size=1
workers=1  ##default

###data loading은 여기에 하시오
###근데 인버전 한 다음에 dataloader에 담는게 더 좋을 것 같다
####몇개 가지고 갖고 놀건진, 여기서 알아서 조정하시오
train_size = 801
train_image_path='./train_images'
files_grab = get_files_frompath(train_image_path, ['*.png', '*.jpg'])
random.shuffle(files_grab)
files_grab = files_grab[0:train_size]
be_tensor = ToTensor()
files_grabbed = []
for i in range(len(files_grab)):
  img = Image.open(files_grab[i])
  img = be_tensor(img)
  files_grabbed.append(img)
#print('data types: ', type(files_grabbed), type(files_grabbed[0]))

#train_list=os.listdir(train_image_path)
#train_dataset = [f for f in train_list if f.endswith('.jpg')]  ### jpg, BMP 등등 알아서 고치기..


##about encoder##
encoder_path = stylegan2_ffhq_1024['e4e_inversion_model']
print('----- Load e4e encoder from {} -----'.format(encoder_path))
ckpt = torch.load(encoder_path, map_location='cpu')
opts = ckpt['opts']
opts['output_size'] = 1024
opts['checkpoint_path'] = encoder_path
opts['device'] = 'cuda'
opts['channel_multiplier'] = stylegan2_ffhq_1024['channel_multiplier']
opts['dataset'] = 'ffhq'              ##maybe the path for
opts = Namespace(**opts)
encoder = pSp(opts)
encoder.cuda().train()

image_resolution=1024   ##an arbitary value.
###stylegan generator
generator_path = stylegan2_ffhq_1024['gan_weights']
channel_multiplier = stylegan2_ffhq_1024['channel_multiplier']
split_sections = stylegan2_ffhq_1024['split_sections']
stylespace_dim = stylegan2_ffhq_1024['stylespace_dim']
exp_ranges = np.load(stylegan2_ffhq_1024['expression_ranges'])
##G is the generator we will use
G = StyleGAN2Generator(image_resolution, 512, 8, channel_multiplier= channel_multiplier)
G.load_state_dict(torch.load(generator_path)['g_ema'], strict = True) 
G.eval()
G.cuda()  ##we will train this generator, so it must be train()


###loading DECA
deca = DECA_model('cuda')
id_loss_ = id_loss.IDLoss().cuda().eval()
lpips_loss = LPIPS(net_type='alex').cuda().eval()
losses = Losses()
#for params in deca.parameters():
    #params.requires_grad=False          ###freezing DECA

###code that takes part in inversion
'''
cropped_image = preprocess_image(source_samples[i], self.fa, save_filename=None)
source_img = image_to_tensor(cropped_image).unsqueeze(0).cuda()
inv_image, source_code = invert_image(source_img, self.encoder, self.G, self.truncation, self.trunc)
'''
train_samples = files_grab
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
inverted_image = []

trunc = G.mean_latent(4096).detach().clone()
truncation = 0.7                       ### default
pil_transfer = ToPILImage()
'''
for i in tqdm(range(len(files_grabbed))):
    cropped_image = preprocess_image(train_samples[i], fa, save_filename=None)
    source_img = image_to_tensor(cropped_image).unsqueeze(0).cuda()
    inv_image, _ = invert_image(source_img, encoder, G, truncation, trunc)
    #print(type(inv_image))
    #print(inv_image.squeeze(0).shape)
    inv_image = pil_transfer(inv_image.squeeze(0))
    inverted_image.append(inv_image)    
#inverted_image=torch.tensor(inverted_image)
#print(type(inverted_image))
'''




from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        
        return data


# Define the transformations

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    #transforms.ToTensor()
    #transforms.CenterCrop(10),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
])

transform2 = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    #transforms.CenterCrop(10),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
])

#print('1st:', len(files_grabbed), len(inverted_image))
# Create an instance of your custom dataset
#files_grabbed = files_grabbed.to('cuda')
#inverted_image = inverted_image.to('cuda')
files_grabbed = CustomDataset(files_grabbed, transform=transform)
#inverted_image = CustomDataset(inverted_image, transform=transform2)
#print('2nd', len(files_grabbed), len(inverted_image))

### tensor 만들기

#files_grabbed = be_tensor(files_grabbed)
#inverted_image = be_tensor(files_grabbed)

'''
inverted_dataset = DataLoader(inverted_image,
                           batch_size=train_batch_size,
                           shuffle=False,                           
                           num_workers=0,
                           drop_last=True)
                           '''
original_set = DataLoader(files_grabbed,
                           batch_size=train_batch_size,
                           shuffle=False,                           
                           num_workers=0,
                           drop_last=True)

###files grabbed is the original asian image
iter_count=0            ##ㅇㄷ서 멈추었는가?
max_iter=10000
lambda_identity=10      ###how much we will look at the identity. default was 10.

### loss caculating func

'''
for i, batch in enumerate(inverted_dataset):
  print(i)
  for img in batch:
    print('this has been inverted')

for i, batch in enumerate(original_set):
  print(i)
  for img in batch:
    print('this is the original image')
'''



def generator_train(iter_count, max_iter, inverted_dataset, original_set):

    ###hyperparams
    lr = 1e-4
    lambda_shape = 1.0
    ###optimizers and device declaration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=5e-2)
    #inverted_dataset = inverted_dataset[0:max_iter]
    #original_set = original_set[0:max_iter]
    loss_list = []
    #loss = 0
    #loss_dict = dict()
    losses = Losses()
    ###batch, original_image in zip(inverted_dataset, original_set)
    for original_image in original_set:
        for ori_img in original_image:
            iter_count+=1
            #print(iter_count)
            cropped_image = preprocess_image(inverted_dataset[iter_count-1], fa, save_filename=None)
            source_img = image_to_tensor(cropped_image).unsqueeze(0).cuda()
            inv_img, _ = invert_image(source_img, encoder, G, truncation, trunc)
            # print(type(inv_image))
            # print(inv_image.squeeze(0).shape)
            #inv_image = pil_transfer(inv_image.squeeze(0))
            #inverted_image.append(inv_image)
            #inv_img = inv_img.unsqueeze(0).cuda()
            ori_img = ori_img.unsqueeze(0).cuda()
            loss_dict = dict()
            loss = 0
            #print('shape of two img', inv_img.shape, ori_img.shape)
            with torch.no_grad():
                #### deca params for inverted image ####
                params_inverted, angle_inverted = calculate_shapemodel(deca, inv_img)
                params_original, angle_original = calculate_shapemodel(deca, ori_img)
                
            #print('modular', (iter_count-1)%5)
            if (iter_count-1) % 100 == 0:
                out_text = '[step {}]'.format(iter_count)
                for key, value in loss_dict.items():
                    out_text += (' | {}: {:.2f}'.format(key, value))
                out_text += '| Mean Loss {:.2f}'.format(np.mean(np.array(loss_list)))
                print(out_text)
                state_dict = {
                    'step': iter_count,
                    'encoder': encoder.state_dict()
                    #'mask_net': G.state_dict()                    
                }
                checkpoint_path = os.path.join("./new_G", 'mask_net_{:06d}.pt'.format(iter_count))
                torch.save(state_dict, checkpoint_path)

            loss, loss_dict = calculate_loss(params_inverted, params_original,lambda_shape, inv_img, ori_img)   ### loss for shape
            loss_list.append(loss.data.item())
            G.zero_grad()
            loss.backward()
            optimizer.step()

           

def calculate_loss(params_inverted, params_original, lambda_shape, inv_img, ori_img):
    loss_dict = dict()
    loss = 0
    coefficients_inv = dict()
    coefficients_inv['pose'] = params_inverted['pose']
    coefficients_inv['exp'] = params_inverted['alpha_exp']
    coefficients_inv['cam'] = params_inverted['cam']
    coefficients_inv['cam'][:, :] = 0.
    coefficients_inv['cam'][:, 0] = 8
    coefficients_inv['shape'] = params_inverted['alpha_shp']
    landmarks2d_inv, landmarks3d_inv, shape_inv = deca.calculate_shape(coefficients_inv)

    coefficients_ori = dict()
    coefficients_ori['pose'] = params_original['pose']
    coefficients_ori['exp'] = params_original['alpha_exp']
    coefficients_ori['cam'] = params_original['cam']
    coefficients_ori['cam'][:, :] = 0.
    coefficients_ori['cam'][:, 0] = 8
    coefficients_ori['shape'] = params_original['alpha_shp']
    landmarks2d_ori, landmarks3d_ori, shape_ori = deca.calculate_shape(coefficients_ori)
    loss_shape = lambda_shape * losses.calculate_shape_loss(shape_inv, shape_ori, normalize=False)
    loss_mouth = lambda_shape * losses.calculate_mouth_loss(landmarks2d_inv, landmarks2d_ori)
    loss_eye = lambda_shape * losses.calculate_eye_loss(landmarks2d_inv, landmarks2d_ori)

    loss_dict['loss_shape'] = loss_shape.data.item()
    loss_dict['loss_eye'] = loss_eye.data.item()
    loss_dict['loss_mouth'] = loss_mouth.data.item()

    #loss_sol_exp = abs(coefficients_ori['alpha_exp']-coefficients_inv['alpha_exp'])
    #loss += loss_sol_exp.item()
    loss += loss_mouth
    loss += loss_shape
    loss += loss_eye

    if lambda_identity != 0:
        loss_identity = lambda_identity * id_loss_(inv_img, ori_img.detach())
        loss_dict['loss_identity'] = loss_identity.data.item()
        loss += loss_identity
    ''''
    걍 원래 잘 안쓰는듯
    if lambda_perceptual != 0:
        inv_img_255 = tensor_to_255(inv_img)
        ori_img_255 = tensor_to_255(ori_img)
        loss_perceptual = lambda_perceptual * lpips_loss(inv_img_255, ori_img_255.detach())
        loss_dict['loss_perceptual'] = loss_perceptual.data.item()
        loss += loss_perceptual
    '''

    loss_dict['loss'] = loss.data.item()

    return loss, loss_dict

generator_train(iter_count, max_iter, train_samples, original_set)   ##train_samples have paths for original image

state_dict = {
                    'step': iter_count,
                    'mask_net': G.state_dict(),
                    'encoder' : encoder.state_dict()
              }
checkpoint_path = os.path.join("./new_G", 'encoder_{:06d}.pt'.format(iter_count))
torch.save(encoder.state_dict(), checkpoint_path)

