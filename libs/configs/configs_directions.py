import os
import numpy as np
ffhq_dict = {

	'yaw_direction':		0,
	'pitch_direction':		1,
	'roll_direction':		-1,
	'jaw_direction':		3,
	'yaw_scale':			40,
	'pitch_scale':			20,
	'roll_scale':			20,
	'ranges_filepath':		'./libs/configs/ranges_FFHQ.npy'
}

def get_direction_ranges(range_filepath):
	
	if os.path.exists(range_filepath):
		exp_ranges = np.load(range_filepath) 
		exp_ranges = np.asarray(exp_ranges).astype('float64')
	else:
		print('{} does not exists'.format(range_filepath))
		exit()
	
	return exp_ranges