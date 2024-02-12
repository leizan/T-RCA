import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

from baseline.easyrca import EasyRCA


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
list_scenarios = ['certain', 'certain_SC', 'uncertain', 'uncertain_SC']
list_sampling_number = [20, 50, 200] # [10, 100, 500, 1000, 2000]
list_num_inters = [2]
list_sig_level = [0.01]
gamma_max = 1

for mechanisme in list_mechanisme:
    for process in list_process:
        for scenario in list_scenarios:
            # complete_final_res = {}
            # simple_final_res = {}
            # for sig_level in list_sig_level:
            #     complete_final_res[str(sig_level)] = {}
            #     simple_final_res[str(sig_level)] = {}
            simple_res_path = os.path.join('..', 'Results_sim_20000', mechanisme, scenario, process + '_EasyRCA.json')
            with open(simple_res_path, 'r') as json_file:
                simple_final_res = json.load(json_file)

            complete_res_path = os.path.join('..', 'Results_com_20000', mechanisme, scenario, process + '_EasyRCA.json')
            with open(complete_res_path, 'r') as json_file:
                complete_final_res = json.load(json_file)
            for sampling_number in list_sampling_number:
                data_folder_path = os.path.join('..', 'RCA_simulated_data', os.path.join(process, scenario), 'historical_data_20000')
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
                        param_data = pd.read_csv(data_path)
                        histo_data_info = os.path.join('..', 'RCA_simulated_data', os.path.join(process, scenario), 'data_info_20000', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                        with open(histo_data_info, 'r') as json_file:
                            histo_data_info = json.load(json_file)
                        param_threshold_dict = histo_data_info['nodes_thres']


                        json_file_path = os.path.join('..', 'RCA_simulated_data', 'graphs', data_path.split('/')[-1].replace('data', 'graph').replace('csv', 'json'))
                        with open(json_file_path, 'r') as json_file:
                            json_graph = json.load(json_file)

                        for num_inter in list_num_inters:
                            if mechanisme == 'one_path':
                                data_info = os.path.join('..', 'RCA_simulated_data', os.path.join(process, scenario), 'data_info_same_path_2_inters_2000', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                            else:
                                data_info = os.path.join('..', 'RCA_simulated_data', os.path.join(process, scenario), 'data_info_2_inters_2000', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                            with open(data_info, 'r') as json_file:
                                data_info = json.load(json_file)
                            true_root_causes = data_info['intervention_node']

                            # Convert the loaded JSON data into a NetworkX graph
                            true_inter_graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])

                            # find root causes
                            normal_node = []
                            pred_root_causes = []
                            if mechanisme == 'one_path':
                                actual_data = pd.read_csv(data_path.replace('historical_data_20000', 'actual_data_same_path_2_inters_2000'))
                            else:
                                actual_data = pd.read_csv(data_path.replace('historical_data_20000', 'actual_data_2_inters_2000'))
                            actual_data = actual_data.head(sampling_number)
                            for node in actual_data.columns:
                                if not (actual_data[node] > param_threshold_dict[node][0]).any():
                                    normal_node.append(node)

                            anomalous_nodes = [i for i in actual_data.columns if i not in normal_node]

                            anomaly_length = actual_data.shape[0]
                            new_actual_data = pd.concat([param_data, actual_data], ignore_index=True)

                            anomalies_start_time = {}
                            for anomalous in anomalous_nodes:
                                anomalies_start_time[anomalous] = param_data.values.shape[0]

                            if 'SL' in scenario:
                                for node in true_inter_graph.nodes:
                                    true_inter_graph.add_edge(node, node)

                            try:
                                erca = EasyRCA(summary_graph=true_inter_graph, anomalous_nodes=anomalous_nodes, anomalies_start_time=anomalies_start_time,
                           anomaly_length=anomaly_length, gamma_max=gamma_max, sig_threshold=sig_level)
                                erca.run(new_actual_data)    # erca.run(actual_data)
                                pred_root_causes_dict = erca.root_causes
                                # pred_root_causes = []
                                # for type in pred_root_causes_dict.keys():
                                #     if len(pred_root_causes_dict[type]['roots']) != 0:
                                #         for cause in pred_root_causes_dict[type]['roots']:
                                #             pred_root_causes.append(cause)
                                pred_root_causes = []
                                for type in pred_root_causes_dict.keys():
                                    if len(pred_root_causes_dict[type]['roots']) != 0:
                                        for cause in pred_root_causes_dict[type]['roots']:
                                            pred_root_causes.append(cause)
                                    if len(pred_root_causes_dict[type]['time_defying']) != 0:
                                        for cause in pred_root_causes_dict[type]['time_defying']:
                                            pred_root_causes.append(cause)
                                    if len(pred_root_causes_dict[type]['data_defying']) != 0:
                                        for cause in pred_root_causes_dict[type]['data_defying']:
                                            pred_root_causes.append(cause)
                                # print('predicted root causes')
                                # print(pred_root_causes)
                            except:
                                pred_root_causes = []

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

            # simple_res_path = os.path.join('..', 'Results_sim_20000', mechanisme, scenario, process + '_EasyRCA.json')
            with open(simple_res_path, 'w') as json_file:
                json.dump(simple_final_res, json_file)

            # complete_res_path = os.path.join('..', 'Results_com_20000', mechanisme, scenario, process + '_EasyRCA.json')
            with open(complete_res_path, 'w') as json_file:
                json.dump(complete_final_res, json_file)
