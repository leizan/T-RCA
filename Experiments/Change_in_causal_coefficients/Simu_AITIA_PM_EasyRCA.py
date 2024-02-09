import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/ama/zanl/pythonProject/RAITIA')
from raitia_raw import RAITIA2011
from baseline.AITIA.Inference import Inference
import networkx as nx


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


gamma_max = 1
# sig_level = 0.1




list_mechanisme = ['EasyRCA']
list_scenarios = ['Parametric'] # ['Parametric', 'Structual']
list_sampling_number = [10, 100, 500, 1000, 2000, 20, 50, 200] # [10, 100, 500, 1000, 2000]
list_num_inters = [2]
list_sig_level = [0.01]
historical_data_length = 20000
normal_ratio = 0.9
abnormal_ratio = 0

gamma_max = 1

for mechanisme in list_mechanisme:
    for scenario in list_scenarios:
        complete_final_res = {}
        simple_final_res = {}
        for sig_level in list_sig_level:
            complete_final_res[str(sig_level)] = {}
            simple_final_res[str(sig_level)] = {}
        for sampling_number in list_sampling_number:
            data_folder_path = os.path.join('..', '..', 'RCA_simulated_data', os.path.join(mechanisme, scenario), 'LOG')
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
                    actual_data = whole_data.head(historical_data_length + sampling_number)

                    histo_data_info = os.path.join('..', '..', 'RCA_simulated_data', os.path.join(mechanisme, scenario), 'data_info', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                    with open(histo_data_info, 'r') as json_file:
                        histo_data_info = json.load(json_file)
                    true_root_causes = histo_data_info["intervention_nodes"]

                    for num_inter in list_num_inters:

                        inference = Inference(actual_data, pb=False)
                        inference.generate_hypotheses_for_effects(causes=inference.alphabet, effects=inference.alphabet)
                        inference.test_for_prima_facie()
                        all_epsilon_averages = inference.calculate_average_epsilons()

                        if len(all_epsilon_averages.keys()) == 0:
                            pref_root_causes = []
                        else:
                            pred_root_causes = []

                            list_edges = []
                            list_epsilons = []
                            for edge in all_epsilon_averages.keys():
                                if all_epsilon_averages[edge] == None:
                                    print('None for ' + str(edge))
                                else:
                                    list_edges.append(edge)
                                    list_epsilons.append(all_epsilon_averages[edge])

                            if len(list_epsilons) == 0:
                                pred_root_causes = []
                            else:
                                list_epsilons = np.array(list_epsilons)
                                if np.std(list_epsilons) == 0:
                                    z_scores = (list_epsilons-np.mean(list_epsilons))
                                else:
                                    z_scores = (list_epsilons-np.mean(list_epsilons))/np.std(list_epsilons)
                                z_scores = np.abs(z_scores)
                                if len(z_scores) <= 2:
                                    for edge in list_edges:
                                        pred_root_causes.append(edge[0])
                                else:
                                    indices_of_largest = np.argsort(z_scores)[-2:]
                                    for i in indices_of_largest:
                                        pred_root_causes.append(list_edges[i][0])

                        pred_root_causes = list(set(pred_root_causes))
                        print('pred_root_causes')
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

        simple_res_path = os.path.join('..', '..', 'Results_sim_20000', mechanisme, scenario, 'AITIA_PM.json')
        with open(simple_res_path, 'w') as json_file:
            json.dump(simple_final_res, json_file)

        complete_res_path = os.path.join('..', '..', 'Results_com_20000', mechanisme, scenario, 'AITIA_PM.json')
        with open(complete_res_path, 'w') as json_file:
            json.dump(complete_final_res, json_file)
