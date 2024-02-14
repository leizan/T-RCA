import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
os.chdir(str(parent))

from baseline.rcd.rcd import top_k_rc

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

name_to_graph = { 'Last_metric_bolt' : 'capacity_last_metric_bolt',
                  'Elastic_search_bolt' : 'capacity_elastic_search_bolt',
                  'Pre-Message dispatcher bolt' : 'pre_Message_dispatcher_bolt',
                  'Check message bolt' : 'check_message_bolt',
                  'Message dispatcher bolt' : 'message_dispatcher_bolt',
                  'Metric bolt' : 'metric_bolt',
                  'Group status information bolt' : 'group_status_information_bolt',
                  'Real time merger bolt' : 'Real_time_merger_bolt'
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
dataFrame = pd.DataFrame()
dict_anomaly = pd.DataFrame()
directoryPath = '../../real_monitoring_data/'

BINS = 5
K = 2

LOCAL_ALPHA = 0.01
DEFAULT_GAMMA = 5

for file_name in glob.glob(directoryPath + '*.csv'):
    if "data_with_incident_between_46683_and_46783" not in file_name:
        col_value = pd.read_csv(file_name, low_memory=False)
        with open(file_name.replace('.csv', '.json')) as json_file:
            x_descri = json.load(json_file)
        dataFrame[name_to_graph[simplify_node_name[x_descri["metric_name"]]]] = col_value['value']
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



anomaly_start = 46683
anomaly_end = 46783
data_start_index = 45683
relative_index_of_accident = anomaly_start - data_start_index
# nb_anomalous_data = anomaly_end - anomaly_start + 1

param_data = dataFrame.loc[data_start_index: anomaly_start-10]
# param_data = dataFrame.loc[:anomaly_end]

# param_data = param_data.loc[:anomaly_start-10]

for col in param_data.columns:
    # mean_value = mean(param_data[col])
    # param_threshold_dict[col] = [ratio_normal*mean_value]
    param_threshold_dict[col] = [np.sort(param_data[col])[int(ratio_normal*param_data[col].shape[0])]]

print(param_threshold_dict)

# discretize the data


for sig_level in sig_level_list:
    res = {}


    for sampling_data in range(10, 110, 10):
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

            actual_data = actual_data.head(sampling_data)
            try:
                output = top_k_rc(normal_df=param_data, anomalous_df=actual_data, k=K,
                                            bins=BINS , gamma=DEFAULT_GAMMA)
                pred_root_causes = output['root_cause']
                # print('pred_root_causes')
                # print(pred_root_causes)
            except:
                pred_root_causes = []

            print('prediction')
            print(pred_root_causes)
            F1 = calculate_F1(pred_root_causes=pred_root_causes, true_root_causes=true_root_causes)
            records.append(F1)
        print('Mean F1: ' + str(np.around(np.mean(records), 2)))
        print('Std F1: ' + str(np.around(np.std(records), 2)))
        res[str(sampling_data)] = (np.around(np.mean(records), 2), np.around(np.std(records), 2))

    res_path = os.path.join('..', '..', 'Results', 'IT_monitoring_data', 'RCD.json')
    with open(res_path, 'w') as json_file:
        json.dump(res, json_file)

