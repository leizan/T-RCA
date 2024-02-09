import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp
from tigramite.independence_tests.gsquared import Gsquared

def TRCA(offline_data, online_data, ts_thresholds, gamma_min, gamma_max, sig_level, TSCG, save_TSCG, save_TSCG_path, know_normal_node, normal_node):
    if TSCG is None:
        # check offline_data, online_data, ts_thresholds are not none
        assert offline_data is not None, 'offline_data is None.'
        assert online_data is not None, 'online_data is None.'
        assert ts_thresholds is not None, 'ts_thresholds is None.'

        #discretize offline_data based on threshold
        for column in offline_data.columns:
            offline_data[column] = offline_data[column].values >= ts_thresholds[column][0]

        # genarate the TSCG
        data_frame = pp.DataFrame(offline_data.values, var_names=offline_data.columns)
        Gsquard_test = Gsquared(significance='analytic')
        pcmci = PCMCI(dataframe=data_frame, cond_ind_test=Gsquard_test, verbosity=-1)

        results = pcmci.run_pcmci(tau_min=gamma_min, tau_max=gamma_max, pc_alpha=sig_level, alpha_level=sig_level)
        # results = pcmci.run_pcmciplus(tau_min=gamma_min, tau_max=gamma_max, pc_alpha=sig_level)
        matrix_graph = results['graph']

        TSCG = nx.DiGraph()
        TSCG.add_nodes_from(offline_data.columns)

        for i in range(len(offline_data.columns)):
            for m in range(len(offline_data.columns)):
                if matrix_graph[i,m,1] == '-->':
                    TSCG.add_edge(offline_data.columns[i], offline_data.columns[m])
                elif matrix_graph[i,m,1] == '<--':
                    TSCG.add_edge(offline_data.columns[m], offline_data.columns[i])

        if save_TSCG:
            assert save_TSCG_path is not None, 'save_TSCG_path is None.'
            graph_info = nx.node_link_data(TSCG)
            with open(save_TSCG_path, "w") as f:
                json.dump(graph_info, f)


    assert sorted(TSCG.nodes) == sorted(online_data.columns), 'TSCG is not correct.'


    pred_root_causes = []
    if not know_normal_node:
        normal_node = []
        for node in online_data.columns:
            if not (online_data[node] > ts_thresholds[node][0]).any():
                normal_node.append(node)

    TSCG_copy = TSCG.copy()

    # Lemma 2
    if len(normal_node) != 0:
        TSCG_copy.remove_nodes_from(normal_node)
    for node in TSCG_copy.nodes:
        parents_of_node = list(TSCG_copy.predecessors(node))
        if len(parents_of_node) == 0:
            pred_root_causes.append(node)
        else:
            if (len(parents_of_node) == 1) and parents_of_node[0] == node:
                pred_root_causes.append(node)

    # Lemma 3
    if len(pred_root_causes) == 0:
        strongly_connected_components = list(nx.strongly_connected_components(TSCG_copy))
        for scc in strongly_connected_components:
            parent_set_scc = []
            for node in scc:
                for ele in TSCG_copy.predecessors(node):
                    parent_set_scc.append(ele)
            parent_set_scc = set(parent_set_scc)
            if parent_set_scc.issubset(scc):
                if len(scc) == 1:
                    pred_root_causes.append(list(scc)[0])
                else:
                    first_anomly_nodes = []
                    for index in range(online_data.values.shape[0]):
                        for node in scc:
                            if online_data[node].values[index] > ts_thresholds[node][0]:
                                first_anomly_nodes.append(node)
                        if len(first_anomly_nodes) != 0:
                            break
                    for node in first_anomly_nodes:
                        pred_root_causes.append(node)

    pred_root_causes = list(set(pred_root_causes))
    return pred_root_causes, TSCG
