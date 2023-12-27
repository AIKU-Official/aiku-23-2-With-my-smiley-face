import os
import numpy as np
import torch
from torchvision import utils as torch_utils
import glob
from datetime import datetime
import json

from libs.utilities.stylespace_utils import encoder, decoder

def make_path(filepath):
	if not os.path.exists(filepath):
		os.makedirs(filepath, exist_ok = True)

def save_arguments_json(args, save_path, filename):
	out_json = os.path.join(save_path, filename)
	# datetime object containing current date and time
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	with open(out_json, 'w') as out:
		stat_dict = args
		json.dump(stat_dict, out)

def get_files_frompath(path, types):
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(os.path.join(path, files)))
	files_grabbed.sort()
	return files_grabbed

def make_noise(batch, dim, truncation=None):
	if isinstance(dim, int):
		dim = [dim]
	if truncation is None or truncation == 1.0:
		return torch.randn([batch] + dim)
	else:
		return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)

def calculate_shapemodel(deca_model, images, image_space = 'gan'):
	img_tmp = images.clone()
	if image_space == 'gan':
		# invert image from [-1,1] to [0,255]
		min_val = -1; max_val = 1
		img_tmp.clamp_(min=min_val, max=max_val)
		img_tmp.add_(-min_val).div_(max_val - min_val + 1e-5)
		img_tmp = img_tmp.mul(255.0)
		
	p_tensor, alpha_shp_tensor, alpha_exp_tensor, angles, cam = deca_model.extract_DECA_params(img_tmp) # params dictionary 
	out_dict = {}
	# file = "ffhq_deca_ear_ortho.pkl" 
	# import pickle
	# with open(file, 'rb') as f:
	# 	ffhq_deca = pickle.load(f)
	# 	ffhq_deca_exp = ffhq_deca['frames']['00000']['exp']
		
	
	# ffhq_deca_exp = torch.tensor(ffhq_deca_exp)

	# ffhq_deca_exp = ffhq_deca_exp.unsqueeze(0)
	out_dict['pose'] = p_tensor
	# out_dict['alpha_exp'] = ffhq_deca_exp.to(p_tensor.device)#alpha_exp_tensor
	out_dict['alpha_exp'] = alpha_exp_tensor
	out_dict['alpha_shp'] = alpha_shp_tensor
	out_dict['cam'] = cam
	
	return out_dict, angles.cuda()

def get_shifted_latent_code(G, z, shift, input_is_latent = False, truncation=1, truncation_latent = None, w_plus = False, num_layers = None):		
	inject_index = G.n_latent
	if not input_is_latent: # Z space
		w = G.get_latent(z)
		latent = w.unsqueeze(1).repeat(1, inject_index, 1)
	else:  # W space 
		latent= z.clone() #z = latent
	# w space
	if not w_plus : #shift  = B x 512
		if num_layers is None : # add shift in all layers
			shift_rep = shift.unsqueeze(1).repeat(1, inject_index, 1)
			latent += shift_rep 
		else: 
			for i in range(num_layers):
				latent[:, i , :] += shift 
	else: #shift  = B x num_layers x 512
		latent[:, :shift.shape[1], :] += shift 
	return latent
  
# def generate_image(G, latent_code, truncation, trunc, image_resolution, split_sections, input_is_latent = False, return_latents = False, resize_image = True):
	
# 	img, _ = G([latent_code], return_latents = return_latents, truncation = truncation, truncation_latent = trunc, input_is_latent = input_is_latent)
# 	style_space, w, noise = encoder(G, latent_code, truncation, trunc, size = image_resolution, input_is_latent = input_is_latent)
# 	if resize_image:
# 		face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))	
# 		img = face_pool(img)

# 	return img, style_space, w, noise
def generate_image(G, latent_code, truncation, trunc, image_resolution, w_plus = True, num_layers_shift=8, shift_code= None,   input_is_latent = False, return_latents = False, resize_image = True):
	if shift_code is None:
		img, _ = G([latent_code], return_latents = return_latents, truncation = truncation, truncation_latent= trunc, input_is_latent = input_is_latent)
	#img, _ = G([latent_code], return_latents = return_latents, truncation = truncation, truncation_latent = trunc, input_is_latent = input_is_latent)
	else : 
		shifted_code = get_shifted_latent_code(G, latent_code, shift_code, input_is_latent=input_is_latent, truncation=truncation,
                                         truncation_latent=trunc, w_plus=w_plus, num_layers=num_layers_shift)
		img, _ = G([shifted_code], return_latents=return_latents, truncation=truncation, truncation_latent=trunc, input_is_latent=input_is_latent)
	
	if resize_image :
		face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		img = face_pool(img)
	if shift_code is None: 
		style_space, w, noise = encoder(G, latent_code, truncation, trunc, size = image_resolution, input_is_latent=input_is_latent)
	else : 
		style_space, w, noise = encoder(G, shifted_code, truncation, trunc, size = image_resolution, input_is_latent=input_is_latent)

	if resize_image:
		face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))	
		img = face_pool(img)

	return img, style_space, w, noise

# def generate_new_stylespace(style_shifted, mask, num_layers_control = None):
# 	if num_layers_control is not None:
# 		new_style_space = style_shifted.clone()
# 		mask_size = mask.shape[1]
# 		new_style_space[:, :mask_size] =  mask * style_shifted[:, :mask_size] 
# 	else:
# 		new_style_space = mask * style_shifted
# 	return new_style_space

def generate_new_stylespace(style_source, style_target, mask, num_layers_control = None):
	if num_layers_control is not None:
		new_style_space = style_source.clone()
		mask_size = mask.shape[1]
		new_style_space[:, :mask_size] =  mask * style_target[:, :mask_size] + (1-mask) * style_source[:, :mask_size]
	else:
		new_style_space = mask * style_target + (1-mask) * style_source
	return new_style_space
def save_image(image, save_image_dir):		
	grid = torch_utils.save_image(
		image,
		save_image_dir,
		normalize=True,
		value_range=(-1, 1),
	)

def save_grid(source_img, target_img, reenacted_img, save_path):
	dim = source_img.shape[2]
	grid_image = torch.zeros(3, dim , 3 * dim)
	grid_image[:, :, :dim] = source_img.squeeze(0)
	grid_image[:, :, dim:dim*2] = target_img.squeeze(0)
	grid_image[:, :, dim*2:] = reenacted_img.squeeze(0)
	save_image(grid_image, save_path)

