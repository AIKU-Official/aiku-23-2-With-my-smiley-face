from torchvision import utils as torch_utils
from torch.utils.data import DataLoader
import imageio 
import os
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
import pickle
from libs.utilities.utils import *
from libs.utilities.image_utils import *
from libs.DECA.estimate_DECA import DECA_model
from libs.models.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.models.mask_predictor import MaskPredictor
from libs.utilities.stylespace_utils import decoder
from libs.configs.config_models import * # stylegan2_ffhq_1024
# from libs.configs.config_directions import *
from libs.criteria.losses import Losses
from libs.criteria import id_loss
from libs.criteria.lpips.lpips import LPIPS
from libs.utilities.utils_inference import generate_grid_image, calculate_evaluation_metrics
from libs.utilities.dataloader import CustomDataset_validation

from libs.utilities.utils_inference import generate_grid_image
from libs.utilities.utils import make_noise 
from libs.utilities.utils import calculate_shapemodel, save_image, generate_image 

from libs.utilities.common_utils import initialize_directions

class Train_Utils(object):
  def __init__(self, params, shape_model, images_dir, images_reenact_dir, truncation, trunc, wandb):
    self.deca = shape_model 
    self.params = params 
    self.images_dir = images_dir 
    self.images_reenact_dir = images_reenact_dir
    self.truncation = truncation
    self.trunc = trunc 
    self.wandb = wandb 
    
  
  def initialize_arguments(self):
    self.output_path = self.params['experiment_path']
    self.use_wandb = self.params['use_wandb']
    self.log_images_wandb = self.params['log_images_wandb']
    self.project_wandb = self.params['project_wandb']
    self.resume_training_model = self.params['resume_training_model']
    
    self.image_resolution = self.params['image_resolution']
    self.dataset_type = self.params['dataset_type']
    self.synthetic_dataset_path = self.params['synthetic_dataset_path']
    self.shift_scale = self.params['shift_scale']
    
    ## For training and evaluation
    self.lr = self.params['lr']
    self.num_layers_control = self.params['num_layers_control']
    self.max_iter = self.params['max_iter']
    self.batch_size = self.params['batch_size']
    self.test_batch_size = self.params['test_batch_size']
    self.workers= self.params['workers']
    
    self.steps_per_log = self.params['steps_per_log']
    self.steps_per_save_models = self.params['steps_per_save_models']
    self.steps_per_evaluation = self.params['steps_per_evaluation']
    self.num_pairs_log = self.params['num_pairs_log']
    self.validation_pairs  = self.params['validation_pairs']
    
    ## Ours ## 
    self.learned_expressions = 15 #self.params['learned_expressions']
    self.shift_scale = 6.0
    self.count_pose, self.num_expressions, self.directions_exp, jaw_dict, self.angle_scales, self.angle_directions = initialize_directions('ffhq',self.learned_expressions,self.shift_scale )
    self.jaw_dict = jaw_dict 
    
  def load_models(self):
    print("-- Load Pretrained Models -- ")
    self.id_loss = id_loss.IDLoss().cuda().eval()
    self.lpips_loss = LPIPS(net_type = 'alex').cuda().eval()
    self.losses = Losses()
  
  def configure_dataset(self):
    self.test_dataset = CustomDataset_validation(synthetic_dataset_path = self.synthetic_dataset_path, validation_pairs = self.validation_pairs)	
    self.test_dataloader = DataLoader(self.test_dataset,
									batch_size=self.test_batch_size ,
									shuffle=False,
									num_workers=int(self.workers),
									drop_last=True)
    
    self.out_dir = self.params['experiment_path']
  
  def make_shift_vector(self, params_source, params_target, angles_source, angles_target, lambda_inter=1.0):
    shift_vector = torch.zeros(self.params['batch_size'], self.learned_expressions).cuda()
    # expression : 50 
    
    jaw_exp_source = params_source['pose'][:, 3]
    jaw_exp_target = params_target['pose'][:, 3]
    # print(jaw_exp_source.shape)
    a = self.jaw_dict['a']
    b = self.jaw_dict['b']
    target = a * jaw_exp_target + b
    source = a * jaw_exp_source + b
    shift_exp = (target - source) * lambda_inter
    shift_vector[:, self.count_pose - 1] = shift_exp[0]
    
    source_exp = params_source['alpha_exp']
    target_exp = params_target['alpha_exp']
    for idx in range(12):
      ind_exp = self.directions_exp[idx]['exp_component']
      target_expression = target_exp[:, ind_exp]
      source_expression = source_exp[:, ind_exp]
      
      a = self.directions_exp[idx]['a']
      b = self.directions_exp[idx]['b']
      target = a * target_expression + b 
      source = a * source_expression + b 
      shifted_exp = target - source
      shift_vector[:, idx + self.count_pose] = shifted_exp[0]    
    return shift_vector
    