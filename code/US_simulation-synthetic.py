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

print(nocross, state)

dict_param = np.load('../data/dict_param_%s.npy' % (state), allow_pickle=True).item()

bipartite = dict_param['poi_cbg_visits_list'][0]
bipartite_normed_product = copy.deepcopy(dict_param['centrality_scores'])

def assortment_continuous_pairs(pairs, vertex_values, node_list_map, weighted = True, SE = False, M = 1):

    top1 = 0.0
    top2 = 0.0
    top3 = 0.0
    total = 0.0

    for i in pairs:
        for j in pairs[i]:
            if i in node_list_map and j in node_list_map:
                top1 += pairs[i][j] * vertex_values[node_list_map[i]] * vertex_values[node_list_map[j]]
                top2 += pairs[i][j] * vertex_values[node_list_map[i]]
                top3 += pairs[i][j] * vertex_values[node_list_map[j]]
                total += pairs[i][j]
    
    return (top1 - (1.0 / total * top2 * top3)) / total


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
    unvax = copy.deepcopy(dict_param['unvax'])
    del dict_param['unvax']
    del dict_param['centrality']
    del dict_param['centrality_scores']
    del dict_param['cbgs_idxs']
    
    

    cbg2idx = dict_param['cbgs_to_idxs']

    centrality_scores_array = np.array(bipartite_normed_product)

    centrality_scores = {ii: centrality_scores_array[ii] for ii in range(len(centrality_scores_array))}
    centrality_scores = list(reversed(sorted(centrality_scores.items(), key=lambda x: x[1])))
    
    vaccine_rate_cbg_e = unvax

    dict_param['poi_cbg_proportions'] = [{1: 0.9}] * bipartite.shape[0]
    dict_param['cbg_day_prop_out'] = None
    
    dict_param['poi_cbg_visits_list'] = [dict_param['poi_cbg_visits_list'][0]] * num_hours
    dict_param['all_hours'] = list(range(num_hours))


    med = .5 
    ori_sizes = copy.deepcopy(dict_param['cbg_sizes'])

    import time
    model_init_kwargs = {}

    from disease_model import Model

    ## order
    vaccine_rate_cbg_inter = 1 - copy.deepcopy(vaccine_rate_cbg_e)
    avg = np.average(vaccine_rate_cbg_inter, weights=ori_sizes)
    if vc == -1:
        vc = avg
    
    vax_all = np.flip(np.sort(vaccine_rate_cbg_inter))

    ii = 0
    for ii in range(0, len(centrality_scores)):
        cbg = centrality_scores[ii][0]
        vaccine_rate_cbg_inter[cbg] = vax_all[ii]
    
    print(vaccine_rate_cbg_inter)

    dict_param['vaccination_rate'] = copy.deepcopy(vaccine_rate_cbg_inter)
    dict_param['vaccination_rate'] -= np.average((vaccine_rate_cbg_inter), weights=ori_sizes) - vc

    dict_param['vaccination_rate'] = np.maximum(0, dict_param['vaccination_rate'])
    dict_param['vaccination_rate'] = np.minimum(1, dict_param['vaccination_rate'])

    
    print(dict_param['vaccination_rate'])
    
    cache['order'] = copy.deepcopy(dict_param['vaccination_rate'])
    dict_param['cbg_idx_groups_to_track'] = copy.deepcopy(cbg_idx_groups_to_track)
    
    if enable == 0 or enable == 6:
        m = Model(
            starting_seed=starting_seed,
            num_seeds=25,
            **model_init_kwargs)

        m.init_exogenous_variables(**dict_param, **exogenous_model_kwargs)
        m.init_endogenous_variables()
        m.simulate_disease_spread(simulate_cases=True, simulate_deaths=True)

        results[keyword]['order'] = copy.deepcopy(m)
        print(np.average(dict_param['vaccination_rate'], weights=ori_sizes))

    ## original
    vaccine_rate_cbg_inter = copy.deepcopy(vaccine_rate_cbg_e)
    dict_param['vaccination_rate'] = 1 - vaccine_rate_cbg_e
            
    avg = np.average(dict_param['vaccination_rate'], weights=ori_sizes)
    if vc == -1:
        vc = avg
    
    dict_param['vaccination_rate'] = dict_param['vaccination_rate'] - avg + vc
    dict_param['vaccination_rate'] *= variance_scale
    dict_param['vaccination_rate'] = np.maximum(0.0, dict_param['vaccination_rate'])
    dict_param['vaccination_rate'] = np.minimum(1.0, dict_param['vaccination_rate'])

    cache['original'] = copy.deepcopy(dict_param['vaccination_rate'])
    dict_param['cbg_idx_groups_to_track'] = copy.deepcopy(cbg_idx_groups_to_track)



    if enable == 0 or enable == 1:
        m = Model(
            starting_seed=starting_seed,
            num_seeds=25,
            **model_init_kwargs)


        m.init_exogenous_variables(**dict_param, **exogenous_model_kwargs)
        m.init_endogenous_variables()
        m.simulate_disease_spread(simulate_cases=True, simulate_deaths=True)

        results[keyword]['original'] = copy.deepcopy(m)
        print(np.average(dict_param['vaccination_rate'], weights=ori_sizes))

    # shuffle
    dict_param['vaccination_rate'] = 1 - vaccine_rate_cbg_e
    np.random.shuffle(dict_param['vaccination_rate'])
    avg = np.average(dict_param['vaccination_rate'], weights=ori_sizes)
    dict_param['vaccination_rate'] = dict_param['vaccination_rate'] - avg + vc
    dict_param['vaccination_rate'] *= variance_scale
    dict_param['vaccination_rate'] = np.maximum(0.0, dict_param['vaccination_rate'])
    dict_param['vaccination_rate'] = np.minimum(1.0, dict_param['vaccination_rate'])

    cache['shuffle'] = copy.deepcopy(dict_param['vaccination_rate'])
    dict_param['cbg_idx_groups_to_track'] = copy.deepcopy(cbg_idx_groups_to_track)


    model_init_kwargs = {}

    from disease_model import Model
    
    if enable == 0 or enable == 4:
        m = Model(
            starting_seed=starting_seed,
            num_seeds=25,
            **model_init_kwargs)


        m.init_exogenous_variables(**dict_param, **exogenous_model_kwargs)
        m.init_endogenous_variables()
        m.simulate_disease_spread(simulate_cases=True, simulate_deaths=True)

        results[keyword]['shuffle'] = copy.deepcopy(m)
        print(np.average(dict_param['vaccination_rate'], weights=ori_sizes))

    # reverse

    import time
    
    if nocross and state == 'all':
        state_codes = [cbg//10000000000 for cbg in dict_param['cbgs_to_idxs']]
        state_code_set = set(state_codes)

        vax_reverse_all = copy.deepcopy(vaccine_rate_cbg_e)

        for state_code in state_code_set:
            vaxs = np.array([(1 - vaccine_rate_cbg_e[i]) for cbg, i in dict_param['cbgs_to_idxs'].items() if cbg // 10000000000 == state_code])
            vaxs_reverse = np.array([1 - (1 - vaccine_rate_cbg_e[i]) for cbg, i in dict_param['cbgs_to_idxs'].items() if cbg // 10000000000 == state_code])
            cbg_sizes = np.array([dict_param['cbg_sizes'][i] for cbg, i in dict_param['cbgs_to_idxs'].items() if cbg // 10000000000 == state_code])
            vaxs_reverse_adjust = vaxs_reverse - np.average(vaxs_reverse, weights=cbg_sizes) + np.average(vaxs, weights=cbg_sizes)

            vaxs_reverse_adjust = np.maximum(0, vaxs_reverse_adjust)
            vaxs_reverse_adjust = np.minimum(1, vaxs_reverse_adjust)

            ii = 0
            for cbg, i in dict_param['cbgs_to_idxs'].items():
                if cbg // 10000000000 == state_code:
                    vax_reverse_all[i] = vaxs_reverse_adjust[ii]
                    ii += 1
                    
        dict_param['vaccination_rate'] = copy.deepcopy(vax_reverse_all)

    else:
        dict_param['vaccination_rate'] = 1 - (1 - vaccine_rate_cbg_e)
        dict_param['vaccination_rate'] -= (np.average(1 -(1 - vaccine_rate_cbg_e), weights=ori_sizes)) - \
                    vc
        dict_param['vaccination_rate'] *= variance_scale

        dict_param['vaccination_rate'] = np.maximum(0, dict_param['vaccination_rate'])
        dict_param['vaccination_rate'] = np.minimum(1, dict_param['vaccination_rate'])

        assert (dict_param['vaccination_rate'] >= 0.0).all()
        assert (dict_param['vaccination_rate'] <= 1.0).all()
    
    dict_param['cbg_idx_groups_to_track'] = copy.deepcopy(cbg_idx_groups_to_track)

    model_init_kwargs = {}

    from disease_model import Model
    
    if enable == 0 or enable == 2:
        m = Model(
            starting_seed=starting_seed,
            num_seeds=25,
            **model_init_kwargs)

        m.init_exogenous_variables(**dict_param, **exogenous_model_kwargs)
        m.init_endogenous_variables()
        m.simulate_disease_spread(simulate_cases=True, simulate_deaths=True)

        results[keyword]['reverse'] = copy.deepcopy(m)
        cache['reverse'] = copy.deepcopy(dict_param['vaccination_rate'])
    
        vaccine_rate_cbg_inter = copy.deepcopy(vaccine_rate_cbg_e)

    ii = 0
    for ii in range(int(0.01 * len(centrality_scores)), len(centrality_scores)):
        cbg = centrality_scores[ii][0]
        if ii % 2 == 0 and ii + 1 < len(centrality_scores):
            cbg2 = centrality_scores[ii+1][0]
            tmp = vaccine_rate_cbg_inter[cbg]
            vaccine_rate_cbg_inter[cbg] = vaccine_rate_cbg_inter[cbg2]
            vaccine_rate_cbg_inter[cbg2] = tmp
        ii += 1

    dict_param['vaccination_rate'] = 1 - vaccine_rate_cbg_inter
    print(np.average(1 - (vaccine_rate_cbg_inter), weights=ori_sizes) - avg)
    dict_param['vaccination_rate'] -= np.average(1 - (vaccine_rate_cbg_inter), weights=ori_sizes) - vc
    dict_param['vaccination_rate'] *= variance_scale

    dict_param['vaccination_rate'] = np.maximum(0, dict_param['vaccination_rate'])
    dict_param['vaccination_rate'] = np.minimum(1, dict_param['vaccination_rate'])

    assert (dict_param['vaccination_rate'] >= 0.0).all()
    assert (dict_param['vaccination_rate'] <= 1.0).all()
    dict_param['cbg_idx_groups_to_track'] = copy.deepcopy(cbg_idx_groups_to_track)

    model_init_kwargs = {}

    from disease_model import Model
    
    if enable == 0 or enable == 3:

        m = Model(
            starting_seed=starting_seed,
            num_seeds=25,
            **model_init_kwargs)

        m.init_exogenous_variables(**dict_param, **exogenous_model_kwargs)
        m.init_endogenous_variables()
        m.simulate_disease_spread(simulate_cases=True, simulate_deaths=True)

        results[keyword]['exchange_ori'] = copy.deepcopy(m)
        cache['exchange_ori'] = copy.deepcopy(dict_param['vaccination_rate'])

#     uniform
    vaccine_rate_cbg_inter = 1 - np.array([vc] * len(vaccine_rate_cbg_e))

    dict_param['vaccination_rate'] = 1 - vaccine_rate_cbg_inter
    dict_param['vaccination_rate'] *= variance_scale
    dict_param['vaccination_rate'] = np.maximum(0, dict_param['vaccination_rate'])
    dict_param['vaccination_rate'] = np.minimum(1, dict_param['vaccination_rate'])

    dict_param['cbg_idx_groups_to_track'] = copy.deepcopy(cbg_idx_groups_to_track)

    model_init_kwargs = {}

    from disease_model import Model

    if enable == 0 or enable == 5:
        m = Model(
            starting_seed=starting_seed,
            num_seeds=25,
            **model_init_kwargs)


        m.init_exogenous_variables(**dict_param, **exogenous_model_kwargs)
        m.init_endogenous_variables()
        m.simulate_disease_spread(simulate_cases=True, simulate_deaths=True)

        results[keyword]['original_uniform'] = copy.deepcopy(m)
        cache['uniform'] = 1 - variance_scale * np.array([np.average(vaccine_rate_cbg_inter, weights=ori_sizes)] * len(vaccine_rate_cbg_e))

    map_idx = {int(k): dict_param['cbgs_to_idxs'][k] for k in dict_param['cbgs_to_idxs']}
    node_list = [int(cbg) for cbg in dict_param['cbgs_to_idxs']]

    results[keyword]['cache'] = cache
    results[keyword]['ori_sizes'] = ori_sizes
    
np.save(
        '../results/results_%s_%d.npy' % 
        (state, enable), 
        results
       )