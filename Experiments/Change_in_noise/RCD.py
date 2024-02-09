import numpy as np
import pandas as pd
import networkx as nx

import os
import json
from tqdm import tqdm

import sys

sys.path.append('/home/lzan/Bureau/Dynamic causal graph/root-cause-analysis/RAITIA/baseline/rcd')
sys.path.append('/home/lzan/Bureau/Dynamic causal graph/root-cause-analysis/RAITIA')

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

from baseline.rcd.rcd import top_k_rc


def cal_precision_recall(ground_truth, predicion):
    pred_num = len(predicion)
    truth_num = len(ground_truth)

    true_pred = 0
    for node in predicion:
        if node in ground_truth:
            true_pred+=1
    if pred_num == 0:
        return (0, true_pred/truth_num)
    else:
        return (true_pred/pred_num, true_pred/truth_num)


causal_graph = nx.DiGraph([('www', 'Website'),
                           ('Auth Service', 'www'),
                           ('API', 'www'),
                           ('Customer DB', 'Auth Service'),
                           ('Customer DB', 'API'),
                           ('Product Service', 'API'),
                           ('Auth Service', 'API'),
                           ('Order Service', 'API'),
                           ('Shipping Cost Service', 'Product Service'),
                           ('Caching Service', 'Product Service'),
                           ('Product DB', 'Caching Service'),
                           ('Customer DB', 'Product Service'),
                           ('Order DB', 'Order Service')])


list_sampling_number = [10, 100, 500, 1000, 2000, 20, 50, 200] #[10, 100, 500, 1000, 2000]
list_num_inters = [2]
list_sig_level = [0.01] # [0.01, 0.05]
list_mechanisme = ['Dowhy_0.5']
list_scenarios = ['different_path', 'one_path']
normal_ratio = 0.9
gamma_max = 1
historical_data_length = 20000
true_inter_graph = causal_graph

gamma_max = 1
BINS = 5
K = 2

LOCAL_ALPHA = 0.01
DEFAULT_GAMMA = 5

for mechanisme in list_mechanisme:
    for scenario in list_scenarios:
        complete_final_res = {}
        simple_final_res = {}
        for sig_level in list_sig_level:
            complete_final_res[str(sig_level)] = {}
            simple_final_res[str(sig_level)] = {}
        for sampling_number in list_sampling_number:
            data_folder_path = os.path.join('..', '..', 'RCA_simulated_data', os.path.join(mechanisme, scenario), 'data')
            data_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if os.path.isfile(os.path.join(data_folder_path, f))]
            res = {}
            for i in list_num_inters:
                res[str(i)] = {}
            for sig_level in list_sig_level:  #np.arange(0.01, 0.2, 0.04).tolist():
                Pre = {}
                Recall = {}
                F1 = {}
                for num_inter in list_num_inters:
                    Pre[str(num_inter)] = []
                    Recall[str(num_inter)] = []
                    F1[str(num_inter)] = []
                for data_path in tqdm(data_files):
                    #establish OSCG based on historical data
                    whole_data = pd.read_csv(data_path)
                    param_data = whole_data.head(historical_data_length)
                    normal_data = param_data.copy(deep=True)
                    actual_data = whole_data.iloc[historical_data_length:historical_data_length+sampling_number].reset_index(drop=True)

                    data_info_path = os.path.join('..', '..', 'RCA_simulated_data', os.path.join(mechanisme, scenario), 'data_info', data_path.split('/')[-1].replace('data', 'data_info').replace('csv', 'json'))
                    with open(data_info_path, 'r') as json_file:
                        data_info = json.load(json_file)
                    true_root_causes = data_info["intervention_nodes"]

                    param_threshold_dict = {}
                    for node in param_data.columns:
                        param_threshold_dict[node] = [np.sort(param_data[node])[int(normal_ratio*param_data[node].shape[0])]]

                    for num_inter in list_num_inters:

                        # find root causes
                        normal_node = []
                        pred_root_causes = []

                        try:
                            output = top_k_rc(normal_df=param_data, anomalous_df=actual_data, k=K,
                                                        bins=BINS, gamma=DEFAULT_GAMMA)
                            pred_root_causes = output['root_cause']
                            print('pred_root_causes')
                            print(pred_root_causes)
                        except:
                            pred_root_causes = []


                        pred_root_causes = list(set(pred_root_causes))
                        print('True toot causes')
                        print(true_root_causes)
                        print('predicted root causes')
                        print(pred_root_causes)
                        pre, recall = cal_precision_recall(ground_truth=true_root_causes, predicion=pred_root_causes)
                        Pre[str(num_inter)].append(pre)
                        Recall[str(num_inter)].append(recall)
                        if pre+recall == 0:
                            F1[str(num_inter)].append(0)
                        else:
                            F1[str(num_inter)].append(2*pre*recall/(pre+recall))

                for num_inter in list_num_inters:
                    res[str(num_inter)][str(sig_level)] = {'MP_SP':(np.round(np.mean(Pre[str(num_inter)]),2), np.round(np.std(Pre[str(num_inter)]),2)),
                                                           'MR_SR':(np.round(np.mean(Recall[str(num_inter)]),2), np.round(np.std(Recall[str(num_inter)]),2)),
                                                           'MF_SF':(np.round(np.mean(F1[str(num_inter)]),2), np.round(np.std(F1[str(num_inter)]),2))}
                    print('Sampling number: ' + str(sampling_number))
                    print('Sig level: '+str(sig_level))
                    # print('precison: ' + str(np.mean(Pre[str(num_inter)])))
                    # print('recall: ' + str(np.mean(Recall[str(num_inter)])))
                    print('mean F1: ' + str(np.mean(F1[str(num_inter)])))
                    print('std F1: ' + str(np.std(F1[str(num_inter)])))

            for num_inter in list_num_inters:
                for sig_level in list_sig_level:
                    complete_final_res[str(sig_level)][str(sampling_number)] = res[str(num_inter)][str(sig_level)]
                    simple_final_res[str(sig_level)][str(sampling_number)] = res[str(num_inter)][str(sig_level)]['MF_SF']

        simple_res_path = os.path.join('..', '..', 'Results_sim_20000', mechanisme, scenario, 'RCD.json')
        with open(simple_res_path, 'w') as json_file:
            json.dump(simple_final_res, json_file)

        complete_res_path = os.path.join('..', '..', 'Results_com_20000', mechanisme, scenario, 'RCD.json')
        with open(complete_res_path, 'w') as json_file:
            json.dump(complete_final_res, json_file)
