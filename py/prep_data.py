import pathlib
import pandas as pd
import numpy as np

# -------------------------
# GENERAL 
# -------------------------

def make_throughput_path():

	# create a throughput folder in which 
	# to hold the data if needed. 

	throughput_path = 'throughput'
	Path(throughput_path).mkdir(exist_ok = True)

def top_n_filter(df, labels = None, top_n = None, col = 'endorsed'):
	'''
	filter df so that only the top n agents by number of endorsements received
	are included. 
	'''
	if top_n is not None:
		the_top = df[[col, 'k']].groupby(col).sum().nlargest(top_n,'k')
		the_top = np.array(the_top.index)
		df = df[df['endorsed'].isin(the_top)]
		df = df[df['endorser'].isin(the_top)]

	first_map = {the_top[i] : str(i) + '_' for i in range(top_n)}
	second_map = {str(i) + '_': i for i in range(top_n)}
	
	df = df.replace({'endorsed' : first_map, 'endorser' : first_map})
	df = df.replace({'endorsed' : second_map, 'endorser' : second_map})

	if labels is not None:
		label_lookup = {second_map[first_map[i]] : labels[i] for i in first_map}

	else: 
		label_lookup = None

	return(df, label_lookup)

def df_to_matrix_sequence(df): 

	n = max(df.endorser.max()+1, df.endorsed.max()+1)

	t_min = df.t.min()
	t_max = df.t.max()

	T_ = np.zeros((t_max - t_min+1, n, n))
	for i in df.index:
		T_[df.t[i] - t_min, df.endorser[i], df.endorsed[i]] += df.k[i] 
	T = np.cumsum(T_, axis = 0) 
	
	return(T)   

def initial_condition(T, timesteps, t_start = 0, t_end = None):
	
	if not t_end:
		t_end = T.shape[0]


	A0 = T[t_start:t_end,:,:].sum(axis = 0)
	A0 = A0 / A0.sum() # normalized

	# mean hiring per year after t_start: 
	v = T[t_start:t_end,:,:].sum(axis = (1,2))
	A0 = A0*((v[-1] - v[0]) / len(v))   

	n_obs = T[-1].sum() - T[t_start].sum()

	return(T[t_start:t_end,:,:], timesteps[t_start:t_end], A0, n_obs)

# -------------------------
# MATH PHD 
# -------------------------

def read_math_phd(path): 
	'''
	current dir is data/PhD Exchange Network Data/
	'''

	# read raw data
	df = pd.read_csv(path + 'PhD_exchange.txt',
		 delim_whitespace=True,
		 names = ('endorsed', 'endorser', 'k', 't'))
	
	df.endorsed -= 1
	df.endorser -= 1
    
	with open(path + 'school_names.txt') as f:
		labels = f.read().splitlines()
		for i in range(len(labels)):
			labels[i] = labels[i].strip()
	
	return(df, labels)

def prep_math_phd(path, top_n = None):

	df, labels = read_math_phd(path)
	df, label_lookup = top_n_filter(df, labels = labels, top_n = top_n)
	T = df_to_matrix_sequence(df)
	timesteps = np.unique(df.t)

	return(T, label_lookup, timesteps)

# -------------------------
# NEWCOMB FRAT 
# -------------------------

def read_newcomb_frat(path):
	M = np.loadtxt(path + 'frat_no_meta.txt')
	return(M)

def prep_newcomb_frat(path, rank_threshold = 1):
	M = read_newcomb_frat(path)

	df = pd.DataFrame(columns=['endorsed', 'endorser',  't', 'k'])

	tau = np.concatenate((np.arange(0,9), np.arange(10, 16)))

	n = M.shape[1]

	for i in range(M.shape[0]):
		for j in range(n):
			if M[i,j] <= rank_threshold: 
				df = df.append(
				{
					'endorser' : i%n,
					'endorsed' : j,
					't' : tau[int(i/n)],
					'k' : 1
				},
				ignore_index = True)

	T = df_to_matrix_sequence(df)

	timesteps = np.arange((tau.max() + 1))

	labels = 'no labels provided'

	return(T, timesteps, labels)

# -------------------------
# PARAKEETS
# -------------------------

def read_parakeets(path): 

	df = pd.read_csv(path + 'aggXquarter.txt', 
			delim_whitespace=True)  

	return(df)

def prep_parakeets(path, group):

	df = read_parakeets(path)
	df = df[df.group == group]

	t_max = df['study.quarter'].max() 
	all_birds = np.unique(np.concatenate((df.actor, df.target)))    

	lookup = {all_birds[i] : i for i in range(len(all_birds))}

	df = df.replace({'actor' : lookup, 'target' : lookup})
	df = df.rename(columns = {'study.quarter' : 't', 
						 'actor' : 'endorsed', 
						 'target' : 'endorser', 
						 'number.wins' : 'k'})

	df = df[['endorser', 'endorsed', 't', 'k']]
	labels = {lookup[key] : key for key in lookup}

	T = df_to_matrix_sequence(df)

	timesteps = np.unique(df.t)

	return(T, timesteps, labels)

# -------------------------
# rt_pol
# -------------------------

def read_rt_pol(path):
	
	df = pd.read_csv(path + 'rt-pol.txt',
					 names = ('endorsed', 'endorser', 't'))
	return(df)

def prep_rt_pol(path, unit = np.timedelta64(1, 'D'), top_n = 50):

	df = read_rt_pol(path)

	df['t'] = pd.to_datetime(pd.to_timedelta(df.t, unit='s'))
	df['t_delta'] = pd.to_timedelta(df.t - df.t.min())

	df['t'] = np.round(df['t_delta'] / unit).astype(int)

	df['k'] = np.ones(len(df.index))

	df = df[['endorsed', 'endorser', 't', 'k']]

	df, labels = top_n_filter(df, labels = None, top_n = top_n)

	T = df_to_matrix_sequence(df)

	timesteps = np.unique(df.t)

	return(T, timesteps)

# -------------------------
# wiki
# -------------------------

def read_wiki(path):
	df = pd.read_csv(path + 'soc-wiki-elec.txt',
					 delimiter = ' ')   
	return(df)

def prep_wiki(path, top_n = 500):

	df = read_wiki(path)

	df['t'] = pd.to_datetime(pd.to_timedelta(df.t, unit='s'))
	df['t_delta'] = pd.to_timedelta(df.t - df.t.min())
	df['t'] = np.round(df['t_delta'] / np.timedelta64(1, 'D')).astype(int)
	df['t'] = (df['t']/28).astype(int)  

	df = df[df.sign > 0]
	df['k'] = np.ones(len(df.index))
	df, labels = top_n_filter(df, top_n = top_n, col = 'endorsed')

	T = df_to_matrix_sequence(df)

	timesteps = np.unique(df.t)

	return(T, timesteps)

def wiki_df(path, top_n = 500):

	df = read_wiki(path)

	df['t'] = pd.to_datetime(pd.to_timedelta(df.t, unit='s'))
	df['t_delta'] = pd.to_timedelta(df.t - df.t.min())
	df['t'] = np.round(df['t_delta'] / np.timedelta64(1, 'D')).astype(int)
	df['t'] = (df['t']/28).astype(int)  

	df = df[df.sign > 0]
	df['k'] = np.ones(len(df.index))
	df, labels = top_n_filter(df, top_n = top_n)

	return(df)

# -------------------------
# chess_transfers
# -------------------------

def read_chess(path):
	df = pd.read_csv(path + 'chess_transfers.csv')  
	return(df)

def prep_chess(path, top_n = 10):

	df = read_chess(path)
	df['t'] = pd.to_datetime(df['Transfer Date'])
	df['t'] = pd.DatetimeIndex(df['t']).year
	
	timesteps = np.unique(df.t)
	
	df['t'] = df['t'] - df['t'].min()
	
	df = df.rename(columns = {
		'Federation' : 'endorsed',
		'Form.Fed'   : 'endorser',
	})
	
	df['endorsed'] = df['endorsed'].astype(str)
	df['endorser'] = df['endorser'].astype(str)
		
	df = df[df.endorser != 'nan']
	df = df[df.endorsed != 'nan']
	
	df['k'] = 1
	
	names = np.unique(np.concatenate((df.endorsed.astype(str), df.endorser.astype(str))))
	recode = {names[i] : i for i in range(len(names))}
	df = df.replace({'endorsed' : recode, 'endorser' : recode})
	
	labels = names.copy()

	df, labels = top_n_filter(df, labels = labels, top_n = top_n)
	
	T = df_to_matrix_sequence(df)
	
	return(T, timesteps, labels)
	
	
	



