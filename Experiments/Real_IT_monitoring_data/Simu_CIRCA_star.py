import os
import sys
# from tigramite.independence_tests.gsquared import Gsquared
from collections import defaultdict
from typing import Dict
from typing import Sequence
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

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

gamma_max = 1
gamma_min = 1
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

true_root_causes = ['Elastic_search_bolt', 'Pre-Message dispatcher bolt']

def dict_to_graph(graph_nx, inter_nodes, str_to_node):
    # Create an empty directed graph
    graph = nx.DiGraph()

    for node in graph_nx.nodes:
        graph.add_node(str_to_node[node])
    for edge in graph_nx.edges:
        graph.add_edge(str_to_node[edge[0]], str_to_node[edge[1]])
    return graph

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
        dict_anomaly[name_to_graph[simplify_node_name[x_descri["metric_name"]]]] = x_descri["anomalies"]


ratio_normal = 0.9 # ideal setting ratio_normal = 1.2 multiple mean can get the best result
# ratio_anomaly = 0
param_threshold_dict = dict()
sig_level_list = [0.01, 0.05]
num_repeat = 1 # 50
# for col in param_data.columns:
#     mean_value = mean(param_data[col])
#     index_anomaly = dict_anomaly[col]
#     mean_anomaly = mean(param_data[col].loc[index_anomaly[0]:index_anomaly[1]])
#     param_threshold_dict[col] = [ratio_normal*mean_value + ratio_anomaly*mean_anomaly]
    # param_threshold_dict[col] = [ratio_normal * mean_value + ratio_anomaly * mean_anomaly, ratio_normal * mean_value]
    # param_threshold_dict[col] = [0.5]



# anomaly_start = 46683
# anomaly_end = 46783
# param_data = dataFrame.iloc[:anomaly_end]

anomaly_start = 46683
anomaly_end = 46783
data_start_index = 45683
relative_index_of_accident = anomaly_start - data_start_index
# nb_anomalous_data = anomaly_end - anomaly_start + 1

param_data = dataFrame.loc[data_start_index: anomaly_start-10]

# param_data = param_data.loc[:anomaly_start-10]

for col in param_data.columns:
    # mean_value = mean(param_data[col])
    # param_threshold_dict[col] = [ratio_normal*mean_value]
    param_threshold_dict[col] = [np.sort(param_data[col])[int(ratio_normal*param_data[col].shape[0])]]

print(param_threshold_dict)



for sig_level in sig_level_list:
    res = {}
    # genarate the OSCG
    data_frame = pp.DataFrame(param_data.values, var_names=param_data.columns) # ,data_type=np.ones(shape=param_data.values.shape)
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
    # 1. Assemble an algorithm
    # circa.graph.common.StaticGraphFactory is derived from circa.graph.GraphFactory
    graph_factory = StaticGraphFactory(MemoryGraph(true_inter_graph))
    scorers = [
        # circa.alg.ci.RHTScorer is derived from circa.alg.common.DecomposableScorer,
        # which is further derived from circa.alg.base.Scorer
        RHTScorer(regressor=ANMRegressor(regressor=LinearRegression())),
    ]
    model = Model(graph_factory=graph_factory, scorers=scorers)


    for sampling_number in range(10, 110, 10):
        records = []
        for index in range(num_repeat):
            # find root causes
            normal_node = []
            pred_root_causes = []
            columns_to_load = ['capacity_last_metric_bolt', 'capacity_elastic_search_bolt', 'pre_Message_dispatcher_bolt',
                           'check_message_bolt', 'message_dispatcher_bolt', 'metric_bolt', 'group_status_information_bolt',
                           'Real_time_merger_bolt']

            for file_name in glob.glob(directoryPath + '*.csv'):
                if "data_with_incident_between_46683_and_46783" in file_name:
                    actual_data = pd.read_csv(file_name, delimiter=';', usecols=columns_to_load)

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
                sli=str_to_node[name_to_graph[true_root_causes[0]]],
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
            # else:
            #     print("Cyclic!!!!!")
            #     pred_root_causes = []


            pred_root_causes = list(set(pred_root_causes))

            print('prediction')
            print(pred_root_causes)
            print('True root causes')
            print([name_to_graph[i] for i in true_root_causes])
            F1 = calculate_F1(pred_root_causes=pred_root_causes, true_root_causes=[name_to_graph[i] for i in true_root_causes])
            records.append(F1)
        print('Mean F1: ' + str(np.around(np.mean(records), 2)))
        print('Std F1: ' + str(np.around(np.std(records), 2)))
        res[str(sampling_number)] = (np.around(np.mean(records), 2), np.around(np.std(records), 2))

    res_path = os.path.join('..', '..', 'Results', 'IT_monitoring_data', 'CIRCA_star.json')
    with open(res_path, 'w') as json_file:
        json.dump(res, json_file)

