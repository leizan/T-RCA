import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from statistics import mean
from matplotlib import pyplot as plt

import sys
sys.path.append('/home/ama/zanl/pythonProject/RAITIA')
from baseline.AITIA.Inference import Inference

gamma_max = 1

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


boolean_variables = []
param_data = pd.DataFrame()
dict_anomaly = pd.DataFrame()
directoryPath = '../../real_monitoring_data/'



for file_name in glob.glob(directoryPath + '*.csv'):
    if "data_with_incident_between_46683_and_46783" not in file_name:
        col_value = pd.read_csv(file_name, low_memory=False)
        with open(file_name.replace('.csv', '.json')) as json_file:
            x_descri = json.load(json_file)
        param_data[simplify_node_name[x_descri["metric_name"]]] = col_value['value']
        dict_anomaly[simplify_node_name[x_descri["metric_name"]]] = x_descri["anomalies"]





ratio_normal = 0.9 # ideal setting ratio_normal = 1.2 multiple mean can get the best result
# ratio_anomaly = 0
param_threshold_dict = dict()
sig_level_list = [0.01] # [0.01, 0.05]
num_repeat = 1 # 50
# for col in param_data.columns:
#     mean_value = mean(param_data[col])
#     index_anomaly = dict_anomaly[col]
#     mean_anomaly = mean(param_data[col].loc[index_anomaly[0]:index_anomaly[1]])
#     param_threshold_dict[col] = [ratio_normal*mean_value + ratio_anomaly*mean_anomaly]
    # param_threshold_dict[col] = [ratio_normal * mean_value + ratio_anomaly * mean_anomaly, ratio_normal * mean_value]
    # param_threshold_dict[col] = [0.5]

actual_data = pd.read_csv('../../monitring_data.csv')

anomaly_start = 46683
anomaly_end = 46783
# param_data = param_data.iloc[:anomaly_end]

# param_data = param_data.loc[:anomaly_start-10]



for sig_level in sig_level_list:
    res = {}
    for sampling_data in range(10, 110, 10):
        records = []
        for index in range(num_repeat):
            actual_data = actual_data.head(anomaly_start + sampling_data)

            inference = Inference(actual_data, pb=False)
            inference.generate_hypotheses_for_effects(causes=inference.alphabet, effects=inference.alphabet)
            inference.test_for_prima_facie()
            all_epsilon_averages = inference.calculate_average_epsilons()

            if len(all_epsilon_averages.keys()) == 0:
                pred_root_causes = []
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
            print('pred root causes')
            print(pred_root_causes)
            F1 = calculate_F1(pred_root_causes=pred_root_causes, true_root_causes=true_root_causes)
            records.append(F1)
        print('Mean F1: ' + str(np.around(np.mean(records), 2)))
        print('Std F1: ' + str(np.around(np.std(records), 2)))
        res[str(sampling_data)] = (np.around(np.mean(records), 2), np.around(np.std(records), 2))

    res_path = os.path.join('..', '..', 'Results_monitoring_data', str(sig_level), 'AITIA_PM.json')
    with open(res_path, 'w') as json_file:
        json.dump(res, json_file)

