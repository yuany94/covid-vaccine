import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vc', type=float, help='fixed vaccination rate', default=0.5)
parser.add_argument('--p_sick_at_t0', type=float, help='p_sick_at_t0', default=0.0001)
parser.add_argument('--home_beta', type=float, help='home_beta', default = 0.000)
parser.add_argument('--poi_psi', type=float, help='poi_psi', default = 1000.0)
parser.add_argument('--num_hours', type=int, help='num_hours', default = 24)
parser.add_argument('--nocross', type=bool, help='no cross', default=False)
parser.add_argument('--state', type=str, help='state', default='all')
parser.add_argument('--enable', type=int, help='enable', default=0)
parser.add_argument('--intervene', type=int, help='intervene', default=0)
parser.add_argument('--distribution', type=str, help='original', default='original')

args = parser.parse_args()

vc = args.vc
p_sick_at_t0 = args.p_sick_at_t0
home_beta = args.home_beta
poi_psi = args.poi_psi
num_hours = args.num_hours
nocross = args.nocross
state = args.state
enable = args.enable
intervene = args.intervene
distribution = args.distribution

print(nocross, state)

if state == 'all':
    dict_param = np.load('../data/dict_param_all_12.npy', allow_pickle=True).item()
else:
    dict_param = np.load('../data/state_dict_param/dict_param_%s.npy' % (state), allow_pickle=True).item()

for k in ['cbgs_idxs', 'centrality_scores', 'unvax']:
    if k in dict_param:
        del dict_param[k]

bipartite = dict_param['poi_cbg_visits_list'][0]

if state == 'all':
    bipartite_normed_product = np.load('../results/centrality_files/bipartite_normed_product_all_12.npy', allow_pickle=True)
else:
    bipartite_normed_product = np.load('../results/centrality_files/bipartite_normed_product_%s.npy' % (state), allow_pickle=True)


import numpy as np
import pandas as pd
from disease_model import Model
import copy
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from scipy.sparse import csr_matrix
import scipy.sparse
import copy
import gc
import argparse

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cbg_idx_groups_to_track = {}
results = {}

starting_seed = 1

state2results = {}
state_name = 'ALL'

for variance_scale in [1.0]:
    cache = {}

    exogenous_model_kwargs = {
         'home_beta': home_beta, 
         'poi_psi': poi_psi,
         'p_sick_at_t0': p_sick_at_t0
        }

    mean_scale = 1.0

    keyword = (vc, p_sick_at_t0, home_beta, poi_psi, num_hours, nocross)
    
    results[keyword] = {}
    unvax = np.ones(bipartite.shape[1]) * 0.5

    cbg2idx = dict_param['cbgs_to_idxs']

    dict_param['poi_cbg_proportions'] = [{1: 0.9}] * bipartite.shape[0]
    dict_param['cbg_day_prop_out'] = None
    
    if state == 'all':
        dict_param['poi_cbg_visits_list'] = [dict_param['poi_cbg_visits_list'][0] / 12 / 12 / 720] * num_hours
    else:
        dict_param['poi_cbg_visits_list'] = [dict_param['poi_cbg_visits_list'][0] / 12 / 12 / 720] * num_hours
    dict_param['all_hours'] = list(range(num_hours))

    med = .5 
    ori_sizes = copy.deepcopy(dict_param['cbg_sizes'])

    import time

    ## original
    if state == 'all':
        vax_distribution = np.load('../results/vax_distribution.npy', allow_pickle=True).item()
    else:
        vax_distribution = np.load('../campaign/vax_distribution_by_state.npy', allow_pickle=True).item()
        vax_distribution = vax_distribution[int(state)]
    if distribution in vax_distribution:
        dict_param['vaccination_rate'] = vax_distribution[distribution]

    if intervene > 0:
        if intervene == 1:
            intervened = np.load('../results/campaign/v.npy', allow_pickle=True)
        elif intervene == 2:
            intervened = np.load('../results/campaign/target_highest_centrality_increase.npy', allow_pickle=True)
        elif intervene == 3:
            intervened = np.load('../results/campaign/target_lowest_vaccination_increase.npy', allow_pickle=True)
        elif intervene == 4:
            intervened = np.load('../results/campaign/target_random_increase.npy', allow_pickle=True)
        elif intervene == 5:
            intervened = np.ones(len(dict_param['vaccination_rate'])) * 0.01
        dict_param['vaccination_rate'] += intervened
    
    
    print(dict_param['vaccination_rate'])
    
    avg = np.average(dict_param['vaccination_rate'], weights=ori_sizes)
    if vc == -1:
        vc = avg
    
    dict_param['vaccination_rate'] = dict_param['vaccination_rate'] - avg + vc
    dict_param['vaccination_rate'] = np.maximum(0.0, dict_param['vaccination_rate'])
    dict_param['vaccination_rate'] = np.minimum(1.0, dict_param['vaccination_rate'])

    cache[distribution] = copy.deepcopy(dict_param['vaccination_rate'])
    dict_param['cbg_idx_groups_to_track'] = copy.deepcopy(cbg_idx_groups_to_track)

    model_init_kwargs = {}

    from disease_model import Model

    m = Model(
        starting_seed=starting_seed,
        num_seeds=25,
        **model_init_kwargs)

    m.init_exogenous_variables(**dict_param, **exogenous_model_kwargs)
    m.init_endogenous_variables()
    m.simulate_disease_spread(simulate_cases=True, simulate_deaths=True)
    
    if num_hours <= 24:
        results[keyword][distribution] = copy.deepcopy(m)
    else:
        results[keyword][distribution] = copy.deepcopy(m.history)
        
    print(np.average(dict_param['vaccination_rate'], weights=ori_sizes))

    np.save(
            '../results/results_%s_%s_%d_intervene_%d_%d_%.2f.npy' % 
            (state, distribution, num_hours, intervene, poi_psi, vc), 
            results
           )