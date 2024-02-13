import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

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

list_mechanisme = ['different_path', 'one_path']
list_process = ['T-DSCM']
list_scenarios = ['uncertain_SC']  # ['certain', 'certain_SC', 'uncertain', 'uncertain_SC']
list_normal_ratio = list(np.array(range(80, 100, 2))/100)
list_normal_ratio.append(-1)
sampling_number = 50
list_num_inters = [2]
list_sig_level = [0.01]
gamma_min = 1
gamma_max = 1

for mechanisme in list_mechanisme:
    for process in list_process:
        for scenario in list_scenarios:
            complete_final_res = {}
            simple_final_res = {}
            for sig_level in list_sig_level:
                complete_final_res[str(sig_level)] = {}
                simple_final_res[str(sig_level)] = {}
            for normal_ratio in list_normal_ratio:
                data_folder_path = os.path.join('../..', 'RCA_simulated_data', os.path.join(process, scenario), 'offline_data')
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
                        param_threshold_dict = {}
                        param_data = pd.read_csv(data_path)
                        histo_data_info = os.path.join('../..', 'RCA_simulated_data', os.path.join(process, scenario), 'offline_data_info', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                        with open(histo_data_info, 'r') as json_file:
                            histo_data_info = json.load(json_file)
                        param_threshold_dict = histo_data_info['nodes_thres']
                        # for node in param_threshold_dict.keys():
                        #     param_threshold_dict[node] = [np.around(param_threshold_dict[node][0] + normal_ratio, 2)]
                        if normal_ratio == -1:
                            param_threshold_dict = histo_data_info['nodes_thres']
                        else:
                            for node in param_data.columns:
                                param_threshold_dict[node] = [np.sort(param_data[node])[int(normal_ratio*param_data[node].shape[0])]]

                        json_file_path = os.path.join('../..', 'RCA_simulated_data', 'graphs', data_path.split('/')[-1].replace('data', 'graph').replace('csv', 'json'))
                        with open(json_file_path, 'r') as json_file:
                            json_graph = json.load(json_file)

                        for num_inter in list_num_inters:
                            if mechanisme == 'one_path':
                                data_info = os.path.join('../..', 'RCA_simulated_data', os.path.join(process, scenario), 'online_data_one_path_info', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                            else:
                                data_info = os.path.join('../..', 'RCA_simulated_data', os.path.join(process, scenario), 'online_data_different_path_info', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                            with open(data_info, 'r') as json_file:
                                data_info = json.load(json_file)
                            true_root_causes = data_info['intervention_node']

                            # Convert the loaded JSON data into a NetworkX graph
                            true_inter_graph = dict_to_graph(graph_dict=json_graph, inter_nodes=true_root_causes)

                            # find root causes
                            normal_node = []
                            if mechanisme == 'one_path':
                                actual_data = pd.read_csv(data_path.replace('offline_data', 'online_data_one_path'))
                            else:
                                actual_data = pd.read_csv(data_path.replace('offline_data', 'online_data_different_path'))
                            actual_data = actual_data.head(sampling_number)

                            pred_root_causes,_ = TRCA(offline_data=param_data, online_data=actual_data, ts_thresholds=param_threshold_dict,
                                                    gamma_min=gamma_min, gamma_max=gamma_max, sig_level=sig_level, TSCG=None, save_TSCG=False, save_TSCG_path=None,
                                                    know_normal_node=False, normal_node=None)

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
            simple_res_path = os.path.join('../..', 'Results_robustness', process, mechanisme, scenario, 'TRCA.json')
            with open(simple_res_path, 'w') as json_file:
                json.dump(simple_final_res, json_file)

