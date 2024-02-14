import json
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
os.chdir(str(parent))

from T_RCA import TRCA


def dict_to_graph(graph_dict, inter_nodes):
    # Create an empty directed graph
    graph = nx.DiGraph()

    # Iterate through the dictionary and add nodes and edges to the graph
    for parent, children in graph_dict.items():
        # Add the parent node to the graph
        graph.add_node(parent)

        # Iterate through the children of the parent
        for child in children.keys():
            # Add the child node to the graph and create a directed edge from parent to child
            graph.add_node(child)
            if child not in inter_nodes:
                graph.add_edge(parent, child)
    return graph

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

list_mechanisme = ['Change_in_causal_coefficients']
list_scenarios = ['one_path']
list_num_inters = [2]
list_sig_level = [0.01]
historical_data_length = 20000
list_normal_ratio = list(np.array(range(80, 100, 2))/100)
sampling_number = 50
gamma_min = 1
gamma_max = 1

for mechanisme in list_mechanisme:
    for scenario in list_scenarios:
        complete_final_res = {}
        simple_final_res = {}
        for sig_level in list_sig_level:
            complete_final_res[str(sig_level)] = {}
            simple_final_res[str(sig_level)] = {}
        for normal_ratio in list_normal_ratio:
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
                    categorical_nodes = []
                    whole_data = pd.read_csv(data_path)
                    param_data = whole_data.head(historical_data_length)
                    actual_data = whole_data.iloc[historical_data_length:historical_data_length+sampling_number].reset_index(drop=True)
                    param_threshold_dict = {}
                    for node in param_data.columns:
                        # param_threshold_dict[node] = [normal_ratio*np.mean(param_data[node].values) + abnormal_ratio*np.mean(actual_data[node].values)]
                        # param_threshold_dict[node] = [normal_ratio*np.mean(param_data[node].values) + abnormal_ratio*np.mean(actual_data[node].values)]
                        param_threshold_dict[node] = [np.sort(param_data[node])[int(normal_ratio*param_data[node].shape[0])]]

                    histo_data_info = os.path.join('..', '..', 'RCA_simulated_data', os.path.join(mechanisme, scenario), 'data_info', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                    with open(histo_data_info, 'r') as json_file:
                        histo_data_info = json.load(json_file)
                    true_root_causes = histo_data_info["intervention_nodes"]

                    json_file_path = os.path.join('..', '..', 'RCA_simulated_data', 'graphs', data_path.split('/')[-1].replace('data', 'graph').replace('csv', 'json'))
                    with open(json_file_path, 'r') as json_file:
                        json_graph = json.load(json_file)

                    for num_inter in list_num_inters:

                        # Convert the loaded JSON data into a NetworkX graph
                        true_inter_graph = dict_to_graph(graph_dict=json_graph, inter_nodes=true_root_causes)

                        true_normal_graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])
                        descendants = set()
                        for node in true_root_causes:
                            descendants |= set(nx.descendants(true_normal_graph, node))

                        descendants = list(descendants)
                        descendants = list(set(descendants +true_root_causes))

                        normal_node = [node for node in actual_data.columns if node not in descendants]

                        pred_root_causes, TSCG = TRCA(offline_data=param_data, online_data=actual_data, ts_thresholds=param_threshold_dict,
                                                    gamma_min=gamma_min, gamma_max=gamma_max, sig_level=sig_level, TSCG=None, save_TSCG=False, save_TSCG_path=None,
                                                     know_normal_node=True, normal_node=normal_node)


                        remain_root_causes = [i for i in true_root_causes if i not in pred_root_causes]
                        descendants_of_roots = []
                        if len(remain_root_causes) != 0:
                            for root in remain_root_causes:
                                if len(nx.descendants(true_inter_graph, root)) != 0:
                                    for i in nx.descendants(true_inter_graph, root):
                                        descendants_of_roots.append(i)
                            descendants_of_roots = list(set(descendants_of_roots))

                            new_normal_node = [i for i in true_inter_graph.nodes if i not in remain_root_causes and i not in descendants_of_roots]

                            TSCG_copy = TSCG.copy()

                            inferred_root_causes, _ = TRCA(offline_data=param_data, online_data=actual_data, ts_thresholds=param_threshold_dict,
                                                gamma_min=gamma_min, gamma_max=gamma_max, sig_level=sig_level, TSCG=TSCG_copy, save_TSCG=False, save_TSCG_path=None,
                                                 know_normal_node=True, normal_node=new_normal_node)

                            for node in inferred_root_causes:
                                pred_root_causes.append(node)


                        pred_root_causes = list(set(pred_root_causes))
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
                    print('Normal ratio: ' + str(normal_ratio))
                    print('Sig level: '+str(sig_level))
                    print('mean F1: ' + str(np.mean(F1[str(num_inter)])))
                    print('std F1: ' + str(np.std(F1[str(num_inter)])))

            for num_inter in list_num_inters:
                for sig_level in list_sig_level:
                    complete_final_res[str(sig_level)][str(normal_ratio)] = res[str(num_inter)][str(sig_level)]
                    simple_final_res[str(sig_level)][str(normal_ratio)] = res[str(num_inter)][str(sig_level)]['MF_SF']

        simple_res_path = os.path.join('..', '..', 'Results_robustness', mechanisme, scenario, 'TRCA_agent.json')
        with open(simple_res_path, 'w') as json_file:
            json.dump(simple_final_res, json_file)

