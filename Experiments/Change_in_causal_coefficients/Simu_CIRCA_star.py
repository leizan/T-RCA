import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from sklearn.linear_model import LinearRegression

from collections import defaultdict
from typing import Dict
from typing import Sequence
from typing import Tuple

from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
os.chdir(str(parent))

from baseline.circa.alg.ci import RHTScorer
from baseline.circa.alg.ci.anm import ANMRegressor
from baseline.circa.alg.common import Model
from baseline.circa.graph.common import StaticGraphFactory
from baseline.circa.model.case import CaseData
from baseline.circa.model.data_loader import MemoryDataLoader
from baseline.circa.model.graph import MemoryGraph
from baseline.circa.model.graph import Node

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp

def dict_to_graph(graph_nx, inter_nodes, str_to_node):
    # Create an empty directed graph
    graph = nx.DiGraph()

    for node in graph_nx.nodes:
        graph.add_node(str_to_node[node])
    for edge in graph_nx.edges:
        graph.add_edge(str_to_node[edge[0]], str_to_node[edge[1]])
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


a = Node("DB", "a")
b = Node("DB", "b")
c = Node("DB", "c")
d = Node("DB", "d")
e = Node("DB", "e")
f = Node("DB", "f")

str_to_node = {'a': a,
               'b': b,
               'c': c,
               'd': d,
               'e': e,
               'f': f}

list_mechanisme = ['Change_in_causal_coefficients']
list_scenarios = ['different_path', 'one_path']
list_sampling_number = [10, 20, 50, 100, 200, 500, 1000, 2000]
list_num_inters = [2]
list_sig_level = [0.01]
historical_data_length = 20000
normal_ratio = 0.9
abnormal_ratio = 0

gamma_min = 1
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
                        param_threshold_dict[node] = [normal_ratio*np.mean(param_data[node].values) + abnormal_ratio*np.mean(actual_data[node].values)]

                    histo_data_info = os.path.join('..', '..', 'RCA_simulated_data', os.path.join(mechanisme, scenario), 'data_info', data_path.split('/')[-1].replace('data', 'info').replace('csv', 'json'))
                    with open(histo_data_info, 'r') as json_file:
                        histo_data_info = json.load(json_file)
                    true_root_causes = histo_data_info["intervention_nodes"]

                    json_file_path = os.path.join('..', '..', 'RCA_simulated_data', 'graphs', data_path.split('/')[-1].replace('data', 'graph').replace('csv', 'json'))
                    with open(json_file_path, 'r') as json_file:
                        json_graph = json.load(json_file)

                    for num_inter in list_num_inters:
                        dataframe = pp.DataFrame(param_data.values,
                                                 datatime=np.arange(len(param_data)),
                                                 var_names=param_data.columns)
                        parcorr = ParCorr(significance='analytic')
                        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

                        output = pcmci.run_pcmciplus(tau_min=gamma_min, tau_max=gamma_max, pc_alpha=sig_level)
                        g = nx.DiGraph()
                        g.add_nodes_from(param_data.columns)
                        for name_y in pcmci.all_parents.keys():
                            for name_x, t_xy in pcmci.all_parents[name_y]:
                                if (param_data.columns[name_x], param_data.columns[name_y]) not in g.edges:
                                    if (param_data.columns[name_y], param_data.columns[name_x]) not in g.edges:
                                        g.add_edges_from([(param_data.columns[name_x], param_data.columns[name_y])])
                        # dag = remove_self_loops(g)
                        # if nx.is_directed_acyclic_graph(dag):
                        true_inter_graph = dict_to_graph(graph_nx=g, inter_nodes=[], str_to_node=str_to_node)

                        graph_factory = StaticGraphFactory(MemoryGraph(true_inter_graph))
                        scorers = [
                            # circa.alg.ci.RHTScorer is derived from circa.alg.common.DecomposableScorer,
                            # which is further derived from circa.alg.base.Scorer
                            RHTScorer(regressor=ANMRegressor(regressor=LinearRegression())),
                        ]
                        model = Model(graph_factory=graph_factory, scorers=scorers)

                        # find root causes
                        normal_node = []
                        pred_root_causes = []
                        for node in actual_data.columns:
                            if not (actual_data[node] > param_threshold_dict[node][0]).any():
                                normal_node.append(node)

                        # anomalous_nodes = [i for i in actual_data.columns if i not in normal_node]
                        # anomalous_nodes = actual_data.columns

                        anomaly_length = actual_data.shape[0]
                        new_actual_data = pd.concat([param_data, actual_data], ignore_index=True)

                        # anomalies_start_time = {}
                        # # for anomalous in anomalous_nodes:
                        # #     anomalies_start_time[anomalous] = param_data.values.shape[0]
                        #
                        # for anomalous in actual_data.columns:
                        #     anomalies_start_time[anomalous] = param_data.values.shape[0]

                        # 2. Prepare data
                        mock_data = {}
                        for node in new_actual_data.columns:
                            mock_data[str_to_node[node]] = list(new_actual_data[node].values)
                        mock_data_with_time: Dict[str, Dict[str, Sequence[Tuple[float, float]]]] = defaultdict(
                            dict
                        )
                        for node, values in mock_data.items():
                            mock_data_with_time[node.entity][node.metric] = [
                                (index * 60, value) for index, value in enumerate(values)
                            ]
                        lookup_window = new_actual_data.values.shape[0]
                        detect_time = 60 * (lookup_window - anomaly_length)
                        data = CaseData(
                            # circa.model.data_loader.MemoryDataLoader is derived from
                            # circa.model.data_loader.DataLoader, which manages data with configurations
                            data_loader=MemoryDataLoader(mock_data_with_time),
                            sli=str_to_node[true_root_causes[0]],
                            detect_time=detect_time,
                            lookup_window=lookup_window,
                            detect_window=sampling_number,
                        )


                        try:
                            pred_root_causes = []
                            dict_root_causes = model.analyze(data=data, current=data.detect_time + 60)
                            # only take the first two nodes
                            pred_root_causes.append(dict_root_causes[0][0].metric)
                            pred_root_causes.append(dict_root_causes[1][0].metric)
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

        simple_res_path = os.path.join('..', '..', 'Results', mechanisme, scenario, 'CIRCA_star.json')
        with open(simple_res_path, 'w') as json_file:
            json.dump(simple_final_res, json_file)

