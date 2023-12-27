import os
import numpy as np
import json
from libs.utilities.image_utils import *
from numpy import ones,vstack
from numpy.linalg import lstsq
from libs.configs.configs_directions import ffhq_dict, get_direction_ranges
def initialize_directions(dataset_type, learned_directions, shift_scale):
	
    angle_scales = np.zeros(3)
    angle_scales[0] = ffhq_dict['yaw_scale']
    angle_scales[1] = ffhq_dict['pitch_scale']
    angle_scales[2] = ffhq_dict['roll_scale']

    angle_directions = np.zeros(3)
    angle_directions[0] = ffhq_dict['yaw_direction']
    angle_directions[1] = ffhq_dict['pitch_direction']
    angle_directions[2] = ffhq_dict['roll_direction']
    exp_ranges = get_direction_ranges(ffhq_dict['ranges_filepath'])
    
    jaw_range = exp_ranges[3]
    jaw_range = jaw_range
    max_jaw = jaw_range[1]
    min_jaw = jaw_range[0]
    exp_ranges = exp_ranges[4:]

    directions_exp = []
    count_pose = 0
    if angle_directions[0] != -1:
        count_pose += 1
    if angle_directions[1] != -1:
        count_pose += 1
    if angle_directions[2] != -1:
        count_pose += 1
    count_pose += 1 # Jaw
    num_expressions = learned_directions - count_pose
	
	
    for i in range(num_expressions):
        dict_3d = {}
        dict_3d['exp_component'] = i
        dict_3d['A_direction'] = i + count_pose 
        dict_3d['max_shift'] =  exp_ranges[i][1]
        dict_3d['min_shift'] =  exp_ranges[i][0] 
        
        points = [(dict_3d['min_shift'], - shift_scale),(dict_3d['max_shift'], shift_scale)]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        dict_3d['a'] = m
        dict_3d['b'] = c
        directions_exp.append(dict_3d)
	
		
    points = [(min_jaw, -6),(max_jaw, 6)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    a_jaw = m
    b_jaw = c

    jaw_dict = {
        'a':			a_jaw,
        'b':			b_jaw,
        'max':			max_jaw,
        'min':			min_jaw
    }

    return count_pose, num_expressions, directions_exp, jaw_dict, angle_scales, angle_directions