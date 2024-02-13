import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import time
from baseline.microcause import micro_cause
from baseline.cloudrange import cloud_ranger
from collections import defaultdict
from typing import Dict
from typing import Sequence
from typing import Tuple
from sklearn.linear_model import LinearRegression

from baseline.circa.alg.ci import RHTScorer
from baseline.circa.alg.ci.anm import ANMRegressor
from baseline.circa.alg.common import Model
from baseline.circa.graph.common import StaticGraphFactory
from baseline.circa.model.case import CaseData
from baseline.circa.model.data_loader import MemoryDataLoader
from baseline.circa.model.graph import MemoryGraph
from baseline.circa.model.graph import Node

from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp
from statistics import mean
from matplotlib import pyplot as plt

from baseline.easyrca import EasyRCA
from baseline.easyrca import remove_self_loops

gamma_max = 1

# res = {}
# sig_level = 0.01



import glob
import json

simplify_node_name = {
    'Real time merger bolt de not_found sur Storm-1': 'Real time merger bolt',
    'Check message bolt de not_found sur storm-1': 'Check message bolt',
    'Message dispatcher bolt de not_found sur storm-1': 'Message dispatcher bolt',
    'Metric bolt de not_found sur Storm-1': 'Metric bolt',
    'Pre-Message dispatcher bolt de not_found sur storm-1': 'Pre-Message dispatcher bolt',
    'capacity_last_metric_bolt de Apache-Storm-bolt_capacity_topology - monitoring_ingestion sur prd-ovh-storm-01': 'Last_metric_bolt',
    'capacity_elastic_search_bolt de Apache-Storm-bolt_capacity_topology - monitoring_ingestion sur prd-ovh-storm-01': 'Elastic_search_bolt',
    'Group status information bolt de not_found sur storm-1': 'Group status information bolt'
}

column_name_transfer = {'capacity_last_metric_bolt': 'Last_metric_bolt',
                        'capacity_elastic_search_bolt': 'Elastic_search_bolt',
                        'pre_Message_dispatcher_bolt': 'Pre-Message dispatcher bolt',
                        'check_message_bolt': 'Check message bolt',
                        'message_dispatcher_bolt': 'Message dispatcher bolt',
                        'metric_bolt': 'Metric bolt',
                        'group_status_information_bolt': 'Group status information bolt',
                        'Real_time_merger_bolt': 'Real time merger bolt'
}

name_to_graph = { 'Last_metric_bolt' : 'capacity_last_metric_bolt',
                  'Elastic_search_bolt' : 'capacity_elastic_search_bolt',
                  'Pre-Message dispatcher bolt' : 'pre_Message_dispatcher_bolt',
                  'Check message bolt' : 'check_message_bolt',
                  'Message dispatcher bolt' : 'message_dispatcher_bolt',
                  'Metric bolt' : 'metric_bolt',
                  'Group status information bolt' : 'group_status_information_bolt',
                  'Real time merger bolt' : 'Real_time_merger_bolt'
}

capacity_last_metric_bolt = Node("DB", "capacity_last_metric_bolt")
capacity_elastic_search_bolt = Node("DB", "capacity_elastic_search_bolt")
pre_Message_dispatcher_bolt = Node("DB", "pre_Message_dispatcher_bolt")
check_message_bolt = Node("DB", "check_message_bolt")
message_dispatcher_bolt = Node("DB", "message_dispatcher_bolts")
metric_bolt = Node("DB", "metric_bolt")
group_status_information_bolt = Node("DB", "group_status_information_bolt")
Real_time_merger_bolt = Node("DB", "Real_time_merger_bolt")

str_to_node = {'capacity_last_metric_bolt': capacity_last_metric_bolt,
               'capacity_elastic_search_bolt': capacity_elastic_search_bolt,
               'pre_Message_dispatcher_bolt': pre_Message_dispatcher_bolt,
               'check_message_bolt': check_message_bolt,
               'message_dispatcher_bolt': message_dispatcher_bolt,
               'metric_bolt': metric_bolt,
               'group_status_information_bolt': group_status_information_bolt,
               'Real_time_merger_bolt': Real_time_merger_bolt}

json_graph = { 'capacity_last_metric_bolt': {},
            'capacity_elastic_search_bolt': {},
            'pre_Message_dispatcher_bolt': {"message_dispatcher_bolt":{} },
            'check_message_bolt': {"Real_time_merger_bolt":{}, "metric_bolt":{}},
            'message_dispatcher_bolt': {"check_message_bolt":{},"Real_time_merger_bolt":{}},
            'metric_bolt': {"capacity_last_metric_bolt":{}},
            'group_status_information_bolt': {"capacity_elastic_search_bolt":{}},
            'Real_time_merger_bolt': {"group_status_information_bolt":{}, "capacity_elastic_search_bolt":{}}}


true_root_causes = ['Elastic_search_bolt', 'Pre-Message dispatcher bolt']

def calculate_F1(pred_root_causes, true_root_causes):
    count_TP = 0
    count_pred = len(pred_root_causes)
    count_true = len(true_root_causes)
    if count_pred != 0:
        for cause in pred_root_causes:
            if cause in true_root_causes:
                count_TP += 1
    if count_pred != 0:
        precision = count_TP/count_pred
    else:
        precision = 0
    recall = count_TP/count_true

    if precision+recall ==0:
        return 0
    else:
        return np.around((2*precision*recall)/(precision+recall), 2)

def dict_to_graph(graph_dict, inter_nodes, str_to_node):
    # Create an empty directed graph
    graph = nx.DiGraph()

    # Iterate through the dictionary and add nodes and edges to the graph
    for parent, children in graph_dict.items():
        # Add the parent node to the graph
        graph.add_node(str_to_node[parent])

        # Iterate through the children of the parent
        for child in children.keys():
            # Add the child node to the graph and create a directed edge from parent to child
            graph.add_node(str_to_node[child])
            if child not in inter_nodes:
                graph.add_edge(str_to_node[parent], str_to_node[child])
    return graph

list_of_methods =  ["EasyRCA", "EasyRCA_star", "MicroCause", "CloudRanger", 'CIRCA']

boolean_variables = []
dataFrame = pd.DataFrame()
dict_anomaly = pd.DataFrame()
directoryPath = '../../real_monitoring_data/'

for file_name in glob.glob(directoryPath + '*.csv'):
    if "data_with_incident_between_46683_and_46783" not in file_name:
        col_value = pd.read_csv(file_name, low_memory=False)
        with open(file_name.replace('.csv', '.json')) as json_file:
            x_descri = json.load(json_file)
        dataFrame[name_to_graph[simplify_node_name[x_descri["metric_name"]]]] = col_value['value']
        dict_anomaly[simplify_node_name[x_descri["metric_name"]]] = x_descri["anomalies"]




anomaly_start = 46683
anomaly_end = 46783
data_start_index = 45683
relative_index_of_accident = anomaly_start - data_start_index
# nb_anomalous_data = anomaly_end - anomaly_start + 1

data = dataFrame.loc[data_start_index:50000]
list_nodes = data.columns
# print(data)

# import graph
graph = nx.DiGraph()
graph.add_nodes_from(list_nodes)
graph.add_edges_from([("pre_Message_dispatcher_bolt", "message_dispatcher_bolt"),
                      ("message_dispatcher_bolt", "check_message_bolt"),
                      ("message_dispatcher_bolt", "Real_time_merger_bolt"),
                      ("check_message_bolt", "Real_time_merger_bolt"),
                      ("check_message_bolt", "metric_bolt"),
                      ("metric_bolt", "capacity_last_metric_bolt"),
                      ("Real_time_merger_bolt", "group_status_information_bolt"),
                      ("Real_time_merger_bolt", "capacity_elastic_search_bolt"),
                      ("group_status_information_bolt", "capacity_elastic_search_bolt")])

for node in graph.nodes:
    graph.add_edge(node, node)

anomalies_start_time = dict()
for node in graph.nodes:
    anomalies_start_time[node] = anomaly_start

sig_level_list = [0.01] # [0.01, 0.05]
num_repeat = 1

start = time.time()
for method in list_of_methods:
    print(method)
    if method == "EasyRCA":
        for sig_level in sig_level_list:
            res = {}
            for nb_anomalous_data in range(10, 110, 10):
                records = []
                for index in range(num_repeat):
                    erca = EasyRCA(graph, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                   anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=sig_level, acyclic_adjustment_set = "ParentsY", adjustment_set = "All")

                    erca.run(data)
                    # print(erca.root_causes)
                    pred_root_causes_dict = erca.root_causes
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
                    pred_root_causes = [column_name_transfer[node] for node in pred_root_causes]
                    # root_causes = []
                    # for subgraph_id in erca.dict_linked_anomalous_graph.keys():
                    #     root_causes = root_causes + erca.root_causes[subgraph_id]["structure_defying"]
                    #     root_causes = root_causes + erca.root_causes[subgraph_id]["structure_defying"]
                    #     root_causes = root_causes + erca.root_causes[subgraph_id]["param_defying"]
                    # print(root_causes)
                    # draw_graph(graph)
                    print(pred_root_causes)
                    F1 = calculate_F1(pred_root_causes=pred_root_causes, true_root_causes=true_root_causes)
                    records.append(F1)
                print('Mean F1: ' + str(np.around(np.mean(records), 2)))
                print('Std F1: ' + str(np.around(np.std(records), 2)))
                res[str(nb_anomalous_data)] = (np.around(np.mean(records), 2), np.around(np.std(records), 2))

            res_path = os.path.join('..', '..', 'Results', 'IT_monitoring_data', method+'.json')
            with open(res_path, 'w') as json_file:
                json.dump(res, json_file)
    elif method == "EasyRCA_star":
        for sig_level in sig_level_list:
            res = {}
            for nb_anomalous_data in range(10, 110, 10):
                records = []
                for index in range(num_repeat):
                    data_normal = data.loc[:anomaly_start - 10]
                    dataframe = pp.DataFrame(data_normal.values,
                                             datatime=np.arange(len(data_normal)),
                                             var_names=data_normal.columns)
                    parcorr = ParCorr(significance='analytic')
                    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

                    output = pcmci.run_pcmciplus(tau_min=0, tau_max=gamma_max, pc_alpha=sig_level)
                    g = nx.DiGraph()
                    g.add_nodes_from(data.columns)
                    g = nx.DiGraph()
                    g.add_nodes_from(data.columns)
                    for name_y in pcmci.all_parents.keys():
                        for name_x, t_xy in pcmci.all_parents[name_y]:
                            if (data.columns[name_x], data.columns[name_y]) not in g.edges:
                                if (data.columns[name_y], data.columns[name_x]) not in g.edges:
                                    g.add_edges_from([(data.columns[name_x], data.columns[name_y])])
                    dag = remove_self_loops(g)
                    if nx.is_directed_acyclic_graph(dag):
                        erca = EasyRCA(g, list(g.nodes), anomalies_start_time=anomalies_start_time,
                                       anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=sig_level)

                        erca.run(data)
                        # print(erca.root_causes)
                        pred_root_causes_dict = erca.root_causes
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
                        pred_root_causes = [column_name_transfer[node] for node in pred_root_causes]
                    else:
                        print("Cyclic!!!!!")
                        pred_root_causes = []
                    print(pred_root_causes)
                    F1 = calculate_F1(pred_root_causes=pred_root_causes, true_root_causes=true_root_causes)
                    records.append(F1)
                print('Mean F1: ' + str(np.around(np.mean(records), 2)))
                print('Std F1: ' + str(np.around(np.std(records), 2)))
                res[str(nb_anomalous_data)] = (np.around(np.mean(records), 2), np.around(np.std(records), 2))

            res_path = os.path.join('..', '..', 'Results', 'IT_monitoring_data', method+'.json')
            with open(res_path, 'w') as json_file:
                json.dump(res, json_file)

    elif method == "MicroCause":
        for sig_level in sig_level_list:
            res = {}
            for nb_anomalous_data in range(10, 110, 10):
                records = []
                for index in range(num_repeat):
                    root_causes = micro_cause(data, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                               anomaly_length=nb_anomalous_data, gamma_max=gamma_max, sig_threshold=sig_level)
                    # print(root_causes)
                    pred_root_causes = [column_name_transfer[node] for node in root_causes]
                    print(pred_root_causes)
                    F1 = calculate_F1(pred_root_causes=pred_root_causes, true_root_causes=true_root_causes)
                    records.append(F1)
                print('Mean F1: ' + str(np.around(np.mean(records), 2)))
                print('Std F1: ' + str(np.around(np.std(records), 2)))
                res[str(nb_anomalous_data)] = (np.around(np.mean(records), 2), np.around(np.std(records), 2))

            res_path = os.path.join('..', '..', 'Results', 'IT_monitoring_data', method+'.json')
            with open(res_path, 'w') as json_file:
                json.dump(res, json_file)
    elif method == "CloudRanger":
        for sig_level in sig_level_list:
            res = {}
            for nb_anomalous_data in range(10, 110, 10):
                records = []
                for index in range(num_repeat):
                    root_causes = cloud_ranger(data, list(graph.nodes), anomalies_start_time=anomalies_start_time,
                                               anomaly_length=nb_anomalous_data, sig_threshold=sig_level)
                    # print(root_causes)
                    pred_root_causes = [column_name_transfer[node] for node in root_causes]
                    print(pred_root_causes)
                    F1 = calculate_F1(pred_root_causes=pred_root_causes, true_root_causes=true_root_causes)
                    records.append(F1)
                print('Mean F1: ' + str(np.around(np.mean(records), 2)))
                print('Std F1: ' + str(np.around(np.std(records), 2)))
                res[str(nb_anomalous_data)] = (np.around(np.mean(records), 2), np.around(np.std(records), 2))

            res_path = os.path.join('..', '..', 'Results', 'IT_monitoring_data', method+'.json')
            with open(res_path, 'w') as json_file:
                json.dump(res, json_file)
    elif method == 'CIRCA':
        for sig_level in sig_level_list:
            res = {}
            for nb_anomalous_data in range(10, 110, 10):
                records = []
                for index in range(num_repeat):
                    true_inter_graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[], str_to_node=str_to_node)
                    for node in true_inter_graph.nodes:
                        true_inter_graph.add_edge(node, node)

                    # 1. Assemble an algorithm
                    # circa.graph.common.StaticGraphFactory is derived from circa.graph.GraphFactory

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



                    new_actual_data = data

                    anomalies_start_time = {}
                    for anomalous in new_actual_data.columns:
                        anomalies_start_time[anomalous] = relative_index_of_accident

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
                    # print(mock_data_with_time)
                    lookup_window = new_actual_data.values.shape[0]
                    detect_time = 60 * relative_index_of_accident
                    case_data = CaseData(
                        # circa.model.data_loader.MemoryDataLoader is derived from
                        # circa.model.data_loader.DataLoader, which manages data with configurations
                        data_loader=MemoryDataLoader(mock_data_with_time),
                        sli=str_to_node['pre_Message_dispatcher_bolt'],
                        detect_time=detect_time,
                        lookup_window=lookup_window,
                        detect_window=nb_anomalous_data,
                    )


                    try:
                        pred_root_causes = []
                        dict_root_causes = model.analyze(data=case_data, current=case_data.detect_time + 60)
                        # only take the first two nodes
                        pred_root_causes.append(column_name_transfer[dict_root_causes[0][0].metric])
                        pred_root_causes.append(column_name_transfer[dict_root_causes[1][0].metric])
                    except:
                        pred_root_causes = []

                    print(pred_root_causes)
                    F1 = calculate_F1(pred_root_causes=pred_root_causes, true_root_causes=true_root_causes)
                    records.append(F1)
                print('Mean F1: ' + str(np.around(np.mean(records), 2)))
                print('Std F1: ' + str(np.around(np.std(records), 2)))
                res[str(nb_anomalous_data)] = (np.around(np.mean(records), 2), np.around(np.std(records), 2))

            res_path = os.path.join('..', '..', 'Results', 'IT_monitoring_data', method+'.json')
            with open(res_path, 'w') as json_file:
                json.dump(res, json_file)
