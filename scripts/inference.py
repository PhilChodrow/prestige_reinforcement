import sys
sys.path.append('py/')

import numpy as np

from utils import *
from prep_data import *
from model import model
from scores import *

from matplotlib import pyplot as plt
import matplotlib

scores = ['SpringRank', 'Root-Degree', 'Eigenvector']

score_functions = {
	'SpringRank' : lambda A: homebrew_SpringRank_score(A.T, alpha = .01),
	'Root-Degree': lambda A: powered_degree_score(A, p = 0.5),
	'Eigenvector': lambda A: eigenvector_score(A.T)
}

# Plotting Params

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['font.sans-serif'] = "Arial Unicode MS"

cset = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']
cset_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD','#000000']

linear_feature = lambda s: np.tile(s, (len(s),1))

def quadratic_feature(s):
	S = linear_feature(s)
	V = (S - S.T)**2
	return(V)



# inference and visualization functions

def inference_suite(control_dict, T, A0):

	models = dict()
	pars = dict()
	for i in range(len(scores)):
		
		score_name = scores[i]

		M = model()
		M.set_score(score_functions[score_name])
		M.set_features([linear_feature, quadratic_feature])
		M.set_data(T, A0)

		# perform maximum-likelihood inference
		par = M.ML(**control_dict[score_name])

		# estimate the Hessian matrix
		H = M.estimate_hessian(par['lam'], par['beta'])
		V = np.linalg.inv(-H)

		# save results
		pars[score_name] = {'par' : par, 'V' : V}
		models[score_name] = M
	return(models, pars)

def block_visualization(T, A0, timesteps, models, pars, selected_model, t_sample, save_path, labels = False, time_unit = None):
	fig, axarr = plt.subplots(2, 3, figsize = (12,8))

	# top row
	model = models[selected_model]
	V = model.GAMMA[t_sample]
	
	ax = axarr[0,0]
	ax.imshow(np.ones_like(V), cmap = 'Greys', vmax = 1, vmin = 0, alpha = 1)
	ax.imshow(matrix_sort(model.A[t_sample], -V.sum(axis = 0)),  cmap = 'Greys', alpha = .99)
	ax.set(title = r'(a) State matrix $\mathbf{A}$') # need to programmatically add the correct timestep in the label
	ax.axis('off')

	ax = axarr[0,1]
	ax.imshow(matrix_sort(V, -V.sum(axis = 0)), alpha = .995, vmax = .4, cmap = 'inferno')
	ax.set(title = '(b) Inferred rate matrix $\mathbf{G}$' +  ' (' + selected_model + ')')
	ax.axis('off')

	ax = axarr[0,2]
	ax.imshow(np.ones_like(T[t_sample]), cmap = 'Greys',vmin = 0, alpha = 1)
	ax.imshow(matrix_sort(T[t_sample+1]-T[t_sample], -V.sum(axis = 0)),   cmap = 'Greys', alpha = .99)
	ax.set(title = '(c) Update $\mathbf{\Delta}$')
	ax.axis('off')
	
	# Get top trajectories

	GAMMA = models[selected_model].GAMMA.mean(axis = 1)
	top_trajectories = np.unique(GAMMA.argmax(axis = 1))
	top_trajectories = (-GAMMA.mean(axis = 0)).argsort()[0:8]

	# lower row
	for p in range(3):
		ax = axarr[1,p]

		score_name = scores[p]
		M = models[score_name]
		par = pars[score_name]

		GAMMA = M.GAMMA.mean(axis = 1)

		g_max = GAMMA.max()

		if score_name == selected_model:

			if p == 0:
				ax.vlines(x = t_sample, ymin = 0, ymax = g_max, linestyle = 'dashed', linewidth = .5)

		ax.plot(timesteps, GAMMA, color = 'grey', alpha = .3)

		

		k = 0
		for i in range(model.n):
			if i in top_trajectories:
				ax.plot(timesteps, GAMMA[:,i], color = cset_muted[k], alpha = 1, linewidth = 3)
				k += 1
		ax.set(ylim = (0, g_max))
		
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# ax.spines['left'].set_bounds(0, GAMMA.max())
		ax.spines['bottom'].set_bounds(timesteps.min(), timesteps.max())

		ax.set(title = score_name)

	if time_unit is not None:
		axarr[1,1].set(xlabel = time_unit)

	plt.tight_layout()

	axarr[1,0].set(ylabel = r'$\gamma$')

	plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)

def parakeet_analysis(group = 'G1', t_start = 0, selected_model = 'SpringRank', t_sample = 2, save_path = 'fig/parakeet_G1_case_study.png'):

	T, timesteps, labels = prep_parakeets('data/parakeet/', group = group)
	T, timesteps, A0, n_obs = initial_condition(T, timesteps, t_start = t_start)

	n = len(labels)

	control_dict = {
		'SpringRank' : {
			'lam0' : 0.8,
			'alpha0' : 10**(-4),
			'tol' : 10**(-4),
			'step_cap' : 0.05,
			'print_updates' : False
		},
		'Root-Degree' : {
			'lam0' : .7, 
			'alpha0' : 10**(-4), 
			'tol' : 10**(-3), 
			'step_cap' : .05,
			'print_updates' : False
		},
		'Eigenvector' : {
			'lam0' : .6, 
			'alpha0' : 10**(-1), 
			'tol' : 10**(-3), 
			'step_cap' : .05,
			'print_updates' : False, 
			'align' : False
		}
	}
	models, pars = inference_suite(control_dict, T, A0)

	block_visualization(T, A0, timesteps, models,  pars, selected_model = 'SpringRank', t_sample = 2, save_path = save_path, labels = False, time_unit = 'Quarter')

if __name__ == '__main__':
	parakeet_analysis(group = 'G1', 
					  t_start=0, 
					  selected_model = 'SpringRank', 
					  t_sample = 2, 
					  save_path = 'fig/parakeet_G1_case_study.png')

	parakeet_analysis(group = 'G2', 
					  t_start=0, 
					  selected_model = 'SpringRank', 
					  t_sample = 2, 
					  save_path = 'fig/parakeet_G2_case_study.png')
	