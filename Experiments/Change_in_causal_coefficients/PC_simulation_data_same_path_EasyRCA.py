import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from raitia import RAITIA2011
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp
from tigramite.independence_tests.gsquared import Gsquared



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

list_mechanisme = ['EasyRCA']
list_scenarios = ['Parametric'] # ['Parametric', 'Structual']
list_sampling_number = [10, 100, 500, 1000, 2000, 20, 50, 200] #[10, 100, 500, 1000, 2000]
list_num_inters = [2]
list_sig_level = [0.01] # [0.01, 0.05]
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

                    # discretize the data
                    for column in param_data.columns:
                        param_data.loc[:, column] = param_data[column].values >= param_threshold_dict[column][0]

                    # genarate the OSCG
                    data_frame = pp.DataFrame(param_data.values, var_names=param_data.columns) # ,data_type=np.ones(shape=param_data.values.shape)
                    Gsquard_test = Gsquared(significance='analytic')
                    pcmci = PCMCI(dataframe=data_frame, cond_ind_test=Gsquard_test, verbosity=-1)

                    results = pcmci.run_pcmci(tau_min=1, tau_max=gamma_max, pc_alpha=sig_level, alpha_level=sig_level)
                    matrix_graph = results['graph']

                    OSCG = nx.DiGraph()
                    OSCG.add_nodes_from(param_data.columns)

                    for i in range(len(param_data.columns)):
                        for m in range(len(param_data.columns)):
                            if matrix_graph[i,m,1] == '-->':
                                OSCG.add_edge(param_data.columns[i], param_data.columns[m])
                            elif matrix_graph[i,m,1] == '<--':
                                OSCG.add_edge(param_data.columns[m], param_data.columns[i])

                    json_file_path = os.path.join('..', '..', 'RCA_simulated_data', 'graphs', data_path.split('/')[-1].replace('data', 'graph').replace('csv', 'json'))
                    with open(json_file_path, 'r') as json_file:
                        json_graph = json.load(json_file)

                    for num_inter in list_num_inters:

                        # Convert the loaded JSON data into a NetworkX graph
                        true_inter_graph = dict_to_graph(graph_dict=json_graph, inter_nodes=true_root_causes)

                        # find root causes
                        normal_node = []
                        pred_root_causes = []
                        # for node in actual_data.columns:
                        #     if not (actual_data[node] > param_threshold_dict[node][0]).any():
                        #         normal_node.append(node)

                        true_normal_graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])
                        descendants = set()
                        for node in true_root_causes:
                            descendants |= set(nx.descendants(true_normal_graph, node))

                        descendants = list(descendants)
                        descendants = list(set(descendants +true_root_causes))

                        normal_node = [node for node in actual_data.columns if node not in descendants]

                        OSCG_copy = OSCG.copy()
                        # print('Abnormal nodes 1')
                        # print([i for i in list(OSCG_copy.nodes) if i not in normal_node])

                        if len(normal_node) != 0:
                            OSCG_copy.remove_nodes_from(normal_node)
                        for node in OSCG_copy.nodes:
                            parents_of_node = list(OSCG_copy.predecessors(node))
                            if len(parents_of_node) == 0:
                                pred_root_causes.append(node)
                            else:
                                if (len(parents_of_node) == 1) and parents_of_node[0] == node:
                                    pred_root_causes.append(node)

                        #########################################################
                        if len(pred_root_causes) == 0:
                            strongly_connected_components = list(nx.strongly_connected_components(OSCG_copy))
                            for scc in strongly_connected_components:
                                parent_set_scc = []
                                for node in scc:
                                    for ele in OSCG_copy.predecessors(node):
                                        parent_set_scc.append(ele)
                                parent_set_scc = set(parent_set_scc)
                                if parent_set_scc.issubset(scc):
                                    if len(scc) == 1:
                                        pred_root_causes.append(scc[0])
                                    else:
                                        first_anomly_nodes = []
                                        for index in range(actual_data.values.shape[0]):
                                            for node in scc:
                                                if actual_data[node].values[index] > param_threshold_dict[node][0]:
                                                    first_anomly_nodes.append(node)
                                            if len(first_anomly_nodes) != 0:
                                                break
                                        for node in first_anomly_nodes:
                                            pred_root_causes.append(node)

                        #########################################################

                        # print('prediction 1')
                        # print(pred_root_causes)
                        if mechanisme == 'EasyRCA':
                            remain_root_causes = [i for i in true_root_causes if i not in pred_root_causes]
                            descendants_of_roots = []
                            if len(remain_root_causes) != 0:
                                for root in remain_root_causes:
                                    if len(nx.descendants(true_inter_graph, root)) != 0:
                                        for i in nx.descendants(true_inter_graph, root):
                                            descendants_of_roots.append(i)
                                descendants_of_roots = list(set(descendants_of_roots))

                                new_normal_node = [i for i in true_inter_graph.nodes if i not in remain_root_causes and i not in descendants_of_roots]

                                OSCG_copy = OSCG.copy()
                                # print('Abnormal nodes 2')
                                # print([i for i in list(OSCG_copy.nodes) if i not in new_normal_node])

                                if len(new_normal_node) != 0:
                                    OSCG_copy.remove_nodes_from(new_normal_node)
                                for node in OSCG_copy.nodes:
                                    parents_of_node = list(OSCG_copy.predecessors(node))
                                    if len(parents_of_node) == 0:
                                        pred_root_causes.append(node)
                                    else:
                                        if (len(parents_of_node) == 1) and parents_of_node[0] == node:
                                            pred_root_causes.append(node)

                                #########################################################
                                if len(pred_root_causes) == 0:
                                    strongly_connected_components = list(nx.strongly_connected_components(OSCG_copy))
                                    for scc in strongly_connected_components:
                                        parent_set_scc = []
                                        for node in scc:
                                            for ele in OSCG_copy.predecessors(node):
                                                parent_set_scc.append(ele)
                                        parent_set_scc = set(parent_set_scc)
                                        if parent_set_scc.issubset(scc):
                                            if len(scc) == 1:
                                                pred_root_causes.append(scc[0])
                                            else:
                                                first_anomly_nodes = []
                                                for index in range(actual_data.values.shape[0]):
                                                    for node in scc:
                                                        if actual_data[node].values[index] > param_threshold_dict[node][0]:
                                                            first_anomly_nodes.append(node)
                                                    if len(first_anomly_nodes) != 0:
                                                        break
                                                for node in first_anomly_nodes:
                                                    pred_root_causes.append(node)

                                    #########################################################

                                # print('prediction 2')
                                # print(pred_root_causes)

                        print('True toot causes')
                        print(true_root_causes)
                        print('predicted root cuases')
                        print(pred_root_causes)

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

        # simple_res_path = os.path.join('..', '..', 'Results_sim_20000', mechanisme, scenario, 'PC.json')   # 'PC_withuot_agent.json'
        # with open(simple_res_path, 'w') as json_file:
        #     json.dump(simple_final_res, json_file)
        #
        # complete_res_path = os.path.join('..', '..', 'Results_com_20000', mechanisme, scenario, 'PC.json') # 'PC_withuot_agent.json'
        # with open(complete_res_path, 'w') as json_file:
        #     json.dump(complete_final_res, json_file)
