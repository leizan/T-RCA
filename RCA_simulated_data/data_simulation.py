import copy
import json
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


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


def visualize_a_graph(graph):
    # Create a layout for the graph (e.g., spring_layout, circular_layout)
    pos = nx.circular_layout(graph)

    # Draw the nodes and edges of the graph
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', edge_color='gray', width=1.0)

    # Display the plot
    plt.show()


def plot_data_with_thres(data, n, dict_nodes_thres):
    plot_data = copy.deepcopy(data)
    plot_data.reset_index(inplace=True)
    for node in data.columns:
        plot_data[node+'_thres'] = np.full(n, dict_nodes_thres[node])
    # visualiser log in time
    fig = go.Figure()

    # Add a trace for each column in the dataframe
    for col in plot_data.columns:
        if col != "index":
            fig.add_trace(go.Scatter(x=plot_data['index'], y=plot_data[col], name=col))

    fig.update_layout(title_x=0.5,
                      title_y=0.9)
    # Finally, we can display the chart using the show function
    fig.show()


def check_impact_of_intervention_node(graph, inter_node, data, dict_nodes_thres, dict_edges_lag, n):
    descendants_list = list(nx.descendants(graph, inter_node))
    index_inter = list(data[data[inter_node] > dict_nodes_thres[inter_node][0]].index)
    for node in descendants_list:
        shortest_path = nx.shortest_path(graph, source=inter_node, target=node)
        lag = 0
        for i in range(len(shortest_path)-1):
            lag+=dict_edges_lag[str((shortest_path[i],shortest_path[i+1]))]
        index_node = [i+lag for i in index_inter if i+lag<n]
        print(node, np.sum(data.loc[index_node][node] < dict_nodes_thres[node][0]))



# graphs_path: path for graphs
# interventions_path: path of the intervention node
# data_path: path to save generated data
# info_path: path to save thresholds of nodes and lags of edges
# n : number of sampling points
# gamma_min: minimum lag
# gamma_max: maximum lag
# thres_min: minimum threshold for all nodes
# thres_max: maximum threshold for all nodes
def generate_historical_data_by_folder_PC(graphs_path, data_path, info_path, n, gamma_min,
                             gamma_max, thres_min, thres_max, prob_inter, epsilon, self_loops=False, max_anomaly=5, seed=3344):
    # np.random.seed(seed=seed)
    self_lag = 1
    if not os.path.exists(data_path):
        # If it doesn't exist, create the folder
        os.makedirs(data_path)
    # check the existence of data information folder
    if not os.path.exists(info_path):
        # If it doesn't exist, create the folder
        os.makedirs(info_path)

    graph_files = [os.path.join(graphs_path, f) for f in os.listdir(graphs_path) if os.path.isfile(os.path.join(graphs_path, f))]

    for json_file_path in tqdm(graph_files):
        with open(json_file_path, 'r') as json_file:
            json_graph = json.load(json_file)

        # Convert the loaded JSON data into a NetworkX graph
        graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])

        nodes_list = list(graph.nodes())
        edges_list = list(graph.edges())
        topological_order = list(nx.topological_sort(graph))

        data = {}
        dict_nodes_thres = {}
        children_list = []
        dict_edges_lag = {}

        for edge in edges_list:
            dict_edges_lag[edge] = np.random.randint(low=gamma_min, high=gamma_max+1)
            children_list.append(edge[1])

        children_list = list(set(children_list))
        for node in nodes_list:
            dict_nodes_thres[node] = [np.round(np.random.uniform(low = thres_min, high = thres_max), 2)]

        for node in nodes_list:
            data[node] = []
            if node not in children_list:
                # data[node] = np.random.uniform(low=0, high=1, size=n)
                for i in range(n):
                    if self_loops and i-self_lag >= 0 and data[node][i-self_lag] >= dict_nodes_thres[node][0] and np.random.uniform() < 1-epsilon:
                        data[node].append(np.random.uniform(low=dict_nodes_thres[node][0], high=1))
                    else:
                        if np.random.uniform() < 1 - prob_inter:
                            data[node].append(np.random.uniform(low=0, high=dict_nodes_thres[node][0]))
                        else:
                            data[node].append(np.random.uniform(low=dict_nodes_thres[node][0], high=1))

                    if i-max_anomaly-1>=0 and np.sum(data[node][i-max_anomaly-1:i-1]>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = np.random.uniform(low=0, high=dict_nodes_thres[node][0])
            else:
                data[node] = np.random.uniform(low=0, high=dict_nodes_thres[node][0], size=n)

        # propagation of interventions
        # for node in topological_order:
        #     values = data[node]
        #     children = list(graph.successors(node))
        #     if len(children) != 0:
        #         for i in range(len(values)):
        #             if values[i] >= dict_nodes_thres[node]:
        #                 for child in children:
        #                     lag = dict_edges_lag[(node,child)]
        #                     if i + lag < n and np.random.uniform() < 1-epsilon:
        #                         data[child][i+lag] = np.random.uniform(low=dict_nodes_thres[child], high=1)
        #     else:
        #          continue

        for node in topological_order:
            parents = list(graph.predecessors(node))
            if len(parents) != 0:
                for i in range(len(data[node])):
                    # abnormal_parent = False
                    abnormal_parent = 0
                    for par in parents:
                        par_values = data[par]
                        lag = dict_edges_lag[(par, node)]
                        if i-lag >=0 and par_values[i-lag] >= dict_nodes_thres[par][0]:
                            # abnormal_parent = True
                            abnormal_parent += 1
                    # if abnormal_parent and np.random.uniform() < 1-epsilon:
                    #     data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)
                    if self_loops and i-self_lag >=0 and data[node][i-self_lag] >= dict_nodes_thres[node][0]:
                        abnormal_parent +=1
                    if abnormal_parent != 0:
                        for m in range(abnormal_parent):
                            if np.random.uniform() < 1-epsilon:
                                data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)
                    else:
                        if np.random.uniform() < prob_inter:
                            data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)

                    if i-max_anomaly-1>=0 and np.sum(data[node][i-max_anomaly-1:i-1]>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = np.random.uniform(low=0, high=dict_nodes_thres[node][0])
            else:
                 continue

        data = pd.DataFrame(data)
        # ## *****************************************************************************************************
        # ## Check data
        # print(json_file_path.split('/')[1].split('.')[0])
        # print("****Check impacts of the intervention node****")
        # print('Intervention node: ' + inter_node)
        # print('Descendants of the intervention node:' + str(list(nx.descendants(graph, inter_node))))
        # check_impact_of_intervention_node(graph=graph, inter_node=inter_node,
        #                                   data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
        # if not only_inter:
        #     print("****Check impacts of the root****")
        #     root_nodes = [i for i in nodes_list if i not in children_list and i != inter_node]
        #     for root in root_nodes:
        #         print('root:' + root)
        #         check_impact_of_intervention_node(graph=graph, inter_node=root,
        #                                   data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
        # ## *****************************************************************************************************
        edges_lag = {}
        for key,value in dict_edges_lag.items():
            edges_lag[str(key)] = value

        info = {'nodes_thres': dict_nodes_thres, 'edges_lag':edges_lag}

        last_node = topological_order[-1]
        print(np.sum(data[last_node]>=dict_nodes_thres[last_node][0])/n)


        data.to_csv(os.path.join(data_path, json_file_path.split('/')[1].replace('graph', 'data').replace('json', 'csv')), index=False)

        data_info_path = os.path.join(info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        # Save the dictionary as a JSON file
        with open(data_info_path, 'w') as json_file:
            json.dump(info, json_file)


def extend_historical_data_by_folder_PC(graphs_path, data_path, info_path, save_info_path, n, gamma_min,
                             gamma_max, thres_min, thres_max, prob_inter, epsilon, self_loops=False, max_anomaly=5, seed=3344):
    # np.random.seed(seed=seed)
    self_lag = 1
    if not os.path.exists(data_path):
        # If it doesn't exist, create the folder
        os.makedirs(data_path)
    # check the existence of data information folder
    if not os.path.exists(save_info_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_info_path)

    graph_files = [os.path.join(graphs_path, f) for f in os.listdir(graphs_path) if os.path.isfile(os.path.join(graphs_path, f))]

    for json_file_path in tqdm(graph_files):
        with open(json_file_path, 'r') as json_file:
            json_graph = json.load(json_file)

        data_info_path = os.path.join(info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        with open(data_info_path, 'r') as json_file:
            data_info = json.load(json_file)

        dict_nodes_thres = data_info['nodes_thres']
        dict_edges_lag = data_info['edges_lag']

        # Convert the loaded JSON data into a NetworkX graph
        graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])

        nodes_list = list(graph.nodes())
        edges_list = list(graph.edges())
        topological_order = list(nx.topological_sort(graph))

        data = {}
        children_list = []

        for edge in edges_list:
            children_list.append(edge[1])

        children_list = list(set(children_list))

        for node in nodes_list:
            data[node] = []
            if node not in children_list:
                # data[node] = np.random.uniform(low=0, high=1, size=n)
                for i in range(n):
                    if self_loops and i-self_lag >= 0 and data[node][i-self_lag] >= dict_nodes_thres[node][0] and np.random.uniform() < 1-epsilon:
                        data[node].append(np.random.uniform(low=dict_nodes_thres[node][0], high=1))
                    else:
                        if np.random.uniform() < 1 - prob_inter:
                            data[node].append(np.random.uniform(low=0, high=dict_nodes_thres[node][0]))
                        else:
                            data[node].append(np.random.uniform(low=dict_nodes_thres[node][0], high=1))

                    if i-max_anomaly-1>=0 and np.sum(np.array(data[node][i-max_anomaly-1:i-1])>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = np.random.uniform(low=0, high=dict_nodes_thres[node][0])
            else:
                data[node] = np.random.uniform(low=0, high=dict_nodes_thres[node][0], size=n)

        # propagation of interventions
        # for node in topological_order:
        #     values = data[node]
        #     children = list(graph.successors(node))
        #     if len(children) != 0:
        #         for i in range(len(values)):
        #             if values[i] >= dict_nodes_thres[node]:
        #                 for child in children:
        #                     lag = dict_edges_lag[(node,child)]
        #                     if i + lag < n and np.random.uniform() < 1-epsilon:
        #                         data[child][i+lag] = np.random.uniform(low=dict_nodes_thres[child], high=1)
        #     else:
        #          continue

        for node in topological_order:
            parents = list(graph.predecessors(node))
            if len(parents) != 0:
                for i in range(len(data[node])):
                    # abnormal_parent = False
                    abnormal_parent = 0
                    for par in parents:
                        par_values = data[par]
                        lag = dict_edges_lag[str((par, node))]
                        if i-lag >=0 and par_values[i-lag] >= dict_nodes_thres[par][0]:
                            # abnormal_parent = True
                            abnormal_parent += 1
                    # if abnormal_parent and np.random.uniform() < 1-epsilon:
                    #     data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)
                    if self_loops and i-self_lag >=0 and data[node][i-self_lag] >= dict_nodes_thres[node][0]:
                        abnormal_parent +=1
                    if abnormal_parent != 0:
                        for m in range(abnormal_parent):
                            if np.random.uniform() < 1-epsilon:
                                data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)
                    else:
                        if np.random.uniform() < prob_inter:
                            data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)

                    if i-max_anomaly-1>=0 and np.sum(data[node][i-max_anomaly-1:i-1]>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = np.random.uniform(low=0, high=dict_nodes_thres[node][0])
            else:
                 continue

        data = pd.DataFrame(data)
        # ## *****************************************************************************************************
        # ## Check data
        # print(json_file_path.split('/')[1].split('.')[0])
        # print("****Check impacts of the intervention node****")
        # print('Intervention node: ' + inter_node)
        # print('Descendants of the intervention node:' + str(list(nx.descendants(graph, inter_node))))
        # check_impact_of_intervention_node(graph=graph, inter_node=inter_node,
        #                                   data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
        # if not only_inter:
        #     print("****Check impacts of the root****")
        #     root_nodes = [i for i in nodes_list if i not in children_list and i != inter_node]
        #     for root in root_nodes:
        #         print('root:' + root)
        #         check_impact_of_intervention_node(graph=graph, inter_node=root,
        #                                   data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
        # ## *****************************************************************************************************


        info = {'nodes_thres': dict_nodes_thres, 'edges_lag': dict_edges_lag}

        last_node = topological_order[-1]
        print(np.sum(data[last_node]>=dict_nodes_thres[last_node][0])/n)

        data.to_csv(os.path.join(data_path, json_file_path.split('/')[1].replace('graph', 'data').replace('json', 'csv')), index=False)

        data_info_path = os.path.join(save_info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        # Save the dictionary as a JSON file
        with open(data_info_path, 'w') as json_file:
            json.dump(info, json_file)



def generate_simulation_data_by_folder_PC(graphs_path, info_path, save_data_path, save_info_path, n, epsilon, only_inter=False, num_inters=1, self_loops=False, max_anomaly=5, in_same_path=False, seed=3344):
    self_lag = 1
    # np.random.seed(seed=seed)
    if not os.path.exists(save_data_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_data_path)

    if not os.path.exists(save_info_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_info_path)

    graph_files = [os.path.join(graphs_path, f) for f in os.listdir(graphs_path) if os.path.isfile(os.path.join(graphs_path, f))]

    for json_file_path in graph_files:
        with open(json_file_path, 'r') as json_file:
            json_graph = json.load(json_file)

        data_info_path = os.path.join(info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        with open(data_info_path, 'r') as json_file:
            data_info = json.load(json_file)

        dict_nodes_thres = data_info['nodes_thres']
        dict_edges_lag = data_info['edges_lag']
        # Convert the loaded JSON data into a NetworkX graph
        graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])
        if num_inters == 1:
            inter_nodes = np.random.choice(graph.nodes, size=1, replace=False)
        else:
            if in_same_path:
                while True:
                    inter_nodes = np.random.choice(graph.nodes, size=num_inters, replace=False)
                    sampled_nodes_in_same_path = True
                    for node_1 in inter_nodes:
                        for node_2 in inter_nodes:
                            if node_1 != node_2:
                                if node_1 not in nx.ancestors(graph, node_2) and node_2 not in nx.ancestors(graph, node_1):
                                    sampled_nodes_in_same_path = False
                    if sampled_nodes_in_same_path:
                        break
            else:
                while True:
                    inter_nodes = np.random.choice(graph.nodes, size=num_inters, replace=False)
                    sampled_nodes_in_same_path = False
                    for node_1 in inter_nodes:
                        for node_2 in inter_nodes:
                            if node_1 != node_2:
                                if node_1 in nx.ancestors(graph, node_2): # and node_2 not in nx.ancestors(graph, node_1)):
                                    sampled_nodes_in_same_path = True
                    if not sampled_nodes_in_same_path:
                        break

        graph = dict_to_graph(graph_dict=json_graph, inter_nodes=inter_nodes)

        nodes_list = list(graph.nodes())
        edges_list = list(graph.edges())
        topological_order = list(nx.topological_sort(graph))

        data = {}
        children_list = []

        for edge in edges_list:
            children_list.append(edge[1])

        children_list = list(set(children_list))


        for node in nodes_list:
            data[node] = []
            if node in inter_nodes:
                data[node] = np.random.uniform(low=dict_nodes_thres[node][0], high=1, size=n)
                # for i in range(n):
                #     if (i+1)%max_anomaly == 1:
                #         data[node][i] = np.random.uniform(low=0, high=dict_nodes_thres[node][0])
            else:
                data[node] = np.random.uniform(low=0, high=dict_nodes_thres[node][0], size=n)


        # transfer interventions
        # for node in topological_order:
        #     values = data[node]
        #     children = list(graph.successors(node))
        #     if len(children) != 0:
        #         for i in range(len(values)):
        #             if values[i] >= dict_nodes_thres[node][0]:
        #                 for child in children:
        #                     lag = dict_edges_lag[str((node,child))]
        #                     if i + lag < n and np.random.uniform() < 1-epsilon:
        #                         data[child][i+lag] = np.random.uniform(low=dict_nodes_thres[child][0], high=1)
        #     else:
        #          continue

        for node in topological_order:
            parents = list(graph.predecessors(node))
            if len(parents) != 0:
                for i in range(len(data[node])):
                    # abnormal_parent = False
                    abnormal_parent = 0
                    for par in parents:
                        par_values = data[par]
                        lag = dict_edges_lag[str((par, node))]
                        if i-lag >=0 and par_values[i-lag] >= dict_nodes_thres[par][0]:
                            # abnormal_parent = True
                            abnormal_parent += 1
                    # if abnormal_parent and np.random.uniform() < 1-epsilon:
                    #     data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)
                    if self_loops and i-self_lag >=0 and data[node][i-self_lag] >= dict_nodes_thres[node][0]:
                        abnormal_parent +=1
                    if abnormal_parent != 0:
                        for m in range(abnormal_parent):
                            if np.random.uniform() < 1-epsilon:
                                data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)

                    if i-max_anomaly-1>=0 and np.sum(data[node][i-max_anomaly-1:i-1]>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = np.random.uniform(low=0, high=dict_nodes_thres[node][0])
            else:
                 continue

        data = pd.DataFrame(data)

        ## *****************************************************************************************************
        ## Check data
        print(json_file_path.split('/')[1].split('.')[0])
        print("****Check impacts of the intervention node****")
        for node in inter_nodes:
            print('Intervention node: ' + node)
            print('Descendants of the intervention node:' + str(list(nx.descendants(graph, node))))
            check_impact_of_intervention_node(graph=graph, inter_node=node,
                                              data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
        if not only_inter:
            print("****Check impacts of the root****")
            root_nodes = [i for i in nodes_list if i not in children_list and i not in inter_nodes]
            for root in root_nodes:
                print('root:' + root)
                check_impact_of_intervention_node(graph=graph, inter_node=root,
                                          data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
        ## *****************************************************************************************************

        info = {'nodes_thres': dict_nodes_thres, 'edges_lag': dict_edges_lag, 'intervention_node': list(inter_nodes)}

        last_node = topological_order[-1]
        print(np.sum(data[last_node]>=dict_nodes_thres[last_node][0])/n)

        data.to_csv(os.path.join(save_data_path, json_file_path.split('/')[1].replace('graph', 'data').replace('json', 'csv')), index=False)

        data_info_path = os.path.join(save_info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        # Save the dictionary as a JSON file
        with open(data_info_path, 'w') as json_file:
            json.dump(info, json_file)


# graphs_path: path for graphs
# save_data_path: path to save generated data
# save_info_path: path to save thresholds of nodes and lags of edges
# n : number of sampling points
# gamma_min: minimum lag
# gamma_max: maximum lag
def generate_historical_data_by_folder_SK(graphs_path, save_data_path, save_info_path, n, thres_min=0.7, thres_max=0.9, gamma_min=1, gamma_max=1, prob_inter=0.3, epsilon=0.3, self_loops=False, max_anomaly=5):
    self_lag = 1
    # np.random.seed(seed=seed)
    if not os.path.exists(save_data_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_data_path)
    # check the existence of data information folder
    if not os.path.exists(save_info_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_info_path)

    #################################################################
    #################################################################
    graph_files = [os.path.join(graphs_path, f) for f in os.listdir(graphs_path) if os.path.isfile(os.path.join(graphs_path, f))]

    for json_file_path in tqdm(graph_files):
        with open(json_file_path, 'r') as json_file:
            json_graph = json.load(json_file)

        # Convert the loaded JSON data into a NetworkX graph
        graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])

        nodes_list = list(graph.nodes())
        edges_list = list(graph.edges())
        topological_order = list(nx.topological_sort(graph))

        data = {}
        dict_nodes_thres = {}
        dict_nodes_base = {}
        coffes_edges = {}
        children_list = []
        dict_edges_lag = {}

        for node in nodes_list:
            dict_nodes_base[node] = [np.round(np.random.uniform(low=0, high=0.1), 2)]
            dict_nodes_thres[node] = [np.round(np.random.uniform(low=thres_min, high=thres_max), 2)]


        for edge in edges_list:
            dict_edges_lag[edge] = np.random.randint(low=gamma_min, high=gamma_max+1)
            coffes_edges[edge] = np.round(np.random.uniform(low=0.1, high=1), 2) #dict_nodes_thres[edge[1]][0]
            children_list.append(edge[1])

        if self_loops:
            for node in nodes_list:
                coffes_edges[(node, node)] = np.round(np.random.uniform(low=0.1, high=1), 2)

        children_list = list(set(children_list))

        for node in nodes_list:
            data[node] = []
            if node not in children_list:
                # data[node] = np.random.uniform(low=0, high=1, size=n)
                for i in range(n):
                    if self_loops and i-self_lag >= 0 and data[node][i-self_lag] >= dict_nodes_thres[node][0] and np.random.uniform() < 1-epsilon:
                        data[node].append(max(dict_nodes_thres[node][0], coffes_edges[(node, node)]*data[node][i-self_lag]+dict_nodes_base[node][0]))
                    else:
                        if np.random.uniform() < 1 - prob_inter:
                            data[node].append(dict_nodes_base[node][0])
                        else:
                            data[node].append(np.random.uniform(low=dict_nodes_thres[node][0], high=dict_nodes_thres[node][0]+1))

                    if i-max_anomaly-1>=0 and np.sum(data[node][i-max_anomaly-1:i-1]>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = dict_nodes_base[node][0]
            else:
                data[node] = [dict_nodes_base[node][0] for i in range(n)]

        # propagation of interventions
        for node in topological_order:
            parents = list(graph.predecessors(node))
            if len(parents) != 0:
                for i in range(len(data[node])):
                    # abnormal_parent = False
                    abnormal_parent = 0
                    for par in parents:
                        par_values = data[par]
                        lag = dict_edges_lag[(par, node)]
                        if i-lag >=0 and par_values[i-lag] >= dict_nodes_thres[par][0]:
                            # abnormal_parent = True
                            abnormal_parent += 1
                    # if abnormal_parent and np.random.uniform() < 1-epsilon:
                    #     data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)
                    if self_loops and i-self_lag >=0 and data[node][i-self_lag] >= dict_nodes_thres[node][0]:
                        abnormal_parent +=1
                    if abnormal_parent != 0:
                        impacted = False
                        for m in range(abnormal_parent):
                            if np.random.uniform() < 1-epsilon:
                                impacted = True # data[node][i] += dict_nodes_thres[node][0]
                        if impacted:
                            for par in parents:
                                lag = dict_edges_lag[(par, node)]
                                if i-lag >=0:
                                    data[node][i]+=coffes_edges[(par, node)]*data[par][i-lag]
                            if self_loops and i-self_lag >=0:
                                data[node][i]+=coffes_edges[(node, node)]*data[node][i-self_lag]
                            data[node][i] = max(dict_nodes_thres[node][0], data[node][i])
                    else:
                        if np.random.uniform() < prob_inter:
                            data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=dict_nodes_thres[node][0]+1)

                    if i-max_anomaly-1>=0 and np.sum(data[node][i-max_anomaly-1:i-1]>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = dict_nodes_base[node][0]
            else:
                 continue

        data = pd.DataFrame(data)

        edges_lag = {}
        for key,value in dict_edges_lag.items():
            edges_lag[str(key)] = value

        edges_coffe = {}
        for key,value in coffes_edges.items():
            edges_coffe[str(key)] = value

        info = {'nodes_thres': dict_nodes_thres, 'nodes_base': dict_nodes_base, 'edges_lag': edges_lag, 'edges_coffe': edges_coffe}

        # last_node = topological_order[-1]
        # print(np.sum(data[last_node]>=dict_nodes_thres[last_node][0])/n)

        data.to_csv(os.path.join(save_data_path, json_file_path.split('/')[1].replace('graph', 'data').replace('json', 'csv')), index=False)

        save_data_info_path = os.path.join(save_info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        # Save the dictionary as a JSON file
        with open(save_data_info_path, 'w') as json_file:
            json.dump(info, json_file)

def extend_historical_data_by_folder_SK(graphs_path, save_data_path, info_path, save_info_path, n, thres_min=0.7, thres_max=0.9, gamma_min=1, gamma_max=1, prob_inter=0.3, epsilon=0.3, self_loops=False, max_anomaly=5):
    self_lag = 1
    # np.random.seed(seed=seed)
    if not os.path.exists(save_data_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_data_path)
    # check the existence of data information folder
    if not os.path.exists(save_info_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_info_path)

    #################################################################
    #################################################################
    graph_files = [os.path.join(graphs_path, f) for f in os.listdir(graphs_path) if os.path.isfile(os.path.join(graphs_path, f))]

    for json_file_path in tqdm(graph_files):
        with open(json_file_path, 'r') as json_file:
            json_graph = json.load(json_file)

        data_info_path = os.path.join(info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        with open(data_info_path, 'r') as json_file:
            data_info = json.load(json_file)

        dict_nodes_thres = data_info['nodes_thres']
        dict_nodes_base = data_info['nodes_base']
        dict_edges_lag = data_info['edges_lag']
        dict_edges_coffe = data_info['edges_coffe']

        # Convert the loaded JSON data into a NetworkX graph
        graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])

        nodes_list = list(graph.nodes())
        edges_list = list(graph.edges())
        topological_order = list(nx.topological_sort(graph))

        data = {}
        children_list = []

        for edge in edges_list:
            children_list.append(edge[1])

        children_list = list(set(children_list))

        for node in nodes_list:
            data[node] = []
            if node not in children_list:
                # data[node] = np.random.uniform(low=0, high=1, size=n)
                for i in range(n):
                    if self_loops and i-self_lag >= 0 and data[node][i-self_lag] >= dict_nodes_thres[node][0] and np.random.uniform() < 1-epsilon:
                        data[node].append(max(dict_nodes_thres[node][0], dict_edges_coffe[str((node, node))]*data[node][i-self_lag]+dict_nodes_base[node][0]))
                    else:
                        if np.random.uniform() < 1 - prob_inter:
                            data[node].append(dict_nodes_base[node][0])
                        else:
                            data[node].append(np.random.uniform(low=dict_nodes_thres[node][0], high=dict_nodes_thres[node][0]+1))

                    if i-max_anomaly-1>=0 and np.sum(np.array(data[node][i-max_anomaly-1:i-1])>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = dict_nodes_base[node][0]
            else:
                data[node] = [dict_nodes_base[node][0] for i in range(n)]

        # propagation of interventions
        for node in topological_order:
            parents = list(graph.predecessors(node))
            if len(parents) != 0:
                for i in range(len(data[node])):
                    # abnormal_parent = False
                    abnormal_parent = 0
                    for par in parents:
                        par_values = data[par]
                        lag = dict_edges_lag[str((par, node))]
                        if i-lag >=0 and par_values[i-lag] >= dict_nodes_thres[par][0]:
                            # abnormal_parent = True
                            abnormal_parent += 1
                    # if abnormal_parent and np.random.uniform() < 1-epsilon:
                    #     data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)
                    if self_loops and i-self_lag >=0 and data[node][i-self_lag] >= dict_nodes_thres[node][0]:
                        abnormal_parent +=1
                    if abnormal_parent != 0:
                        impacted = False
                        for m in range(abnormal_parent):
                            if np.random.uniform() < 1-epsilon:
                                impacted = True # data[node][i] += dict_nodes_thres[node][0]
                        if impacted:
                            for par in parents:
                                lag = dict_edges_lag[str((par, node))]
                                if i-lag >=0:
                                    data[node][i]+=dict_edges_coffe[str((par, node))]*data[par][i-lag]
                            if self_loops and i-self_lag >=0:
                                data[node][i]+=dict_edges_coffe[str((node, node))]*data[node][i-self_lag]
                            data[node][i] = max(dict_nodes_thres[node][0], data[node][i])
                    else:
                        if np.random.uniform() < prob_inter:
                            data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=dict_nodes_thres[node][0]+1)

                    if i-max_anomaly-1>=0 and np.sum(np.array(data[node][i-max_anomaly-1:i-1])>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = dict_nodes_base[node][0]
            else:
                 continue

        data = pd.DataFrame(data)

        info = {'nodes_thres': dict_nodes_thres, 'nodes_base': dict_nodes_base, 'edges_lag': dict_edges_lag, 'edges_coffe': dict_edges_coffe}

        # last_node = topological_order[-1]
        # print(np.sum(data[last_node]>=dict_nodes_thres[last_node][0])/n)

        data.to_csv(os.path.join(save_data_path, json_file_path.split('/')[1].replace('graph', 'data').replace('json', 'csv')), index=False)

        save_data_info_path = os.path.join(save_info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        # Save the dictionary as a JSON file
        with open(save_data_info_path, 'w') as json_file:
            json.dump(info, json_file)

def generate_simulation_data_by_folder_SK(graphs_path, info_path, save_data_path, save_info_path, n, num_inters=1, epsilon=0.3, self_loops=False, max_anomaly=5, in_same_path=False):
    self_lag = 1
    # np.random.seed(seed=seed)
    if not os.path.exists(save_data_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_data_path)

    if not os.path.exists(save_info_path):
        # If it doesn't exist, create the folder
        os.makedirs(save_info_path)

    graph_files = [os.path.join(graphs_path, f) for f in os.listdir(graphs_path) if os.path.isfile(os.path.join(graphs_path, f))]

    for json_file_path in tqdm(graph_files):
        with open(json_file_path, 'r') as json_file:
            json_graph = json.load(json_file)

        data_info_path = os.path.join(info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        with open(data_info_path, 'r') as json_file:
            data_info = json.load(json_file)

        dict_nodes_thres = data_info['nodes_thres']
        dict_nodes_base = data_info['nodes_base']
        dict_edges_lag = data_info['edges_lag']
        dict_edges_coffe = data_info['edges_coffe']
        # Convert the loaded JSON data into a NetworkX graph
        graph = dict_to_graph(graph_dict=json_graph, inter_nodes=[])
        if num_inters == 1:
            inter_nodes = np.random.choice(graph.nodes, size=1, replace=False)
        else:
            if in_same_path:
                while True:
                    inter_nodes = np.random.choice(graph.nodes, size=num_inters, replace=False)
                    sampled_nodes_in_same_path = True
                    for node_1 in inter_nodes:
                        for node_2 in inter_nodes:
                            if node_1 != node_2:
                                if node_1 not in nx.ancestors(graph, node_2) and node_2 not in nx.ancestors(graph, node_1):
                                    sampled_nodes_in_same_path = False
                    if sampled_nodes_in_same_path:
                        break
            else:
                while True:
                    inter_nodes = np.random.choice(graph.nodes, size=num_inters, replace=False)
                    sampled_nodes_in_same_path = False
                    for node_1 in inter_nodes:
                        for node_2 in inter_nodes:
                            if node_1 != node_2:
                                if node_1 in nx.ancestors(graph, node_2): # and node_2 not in nx.ancestors(graph, node_1)):
                                    sampled_nodes_in_same_path = True
                    if not sampled_nodes_in_same_path:
                        break

        graph = dict_to_graph(graph_dict=json_graph, inter_nodes=inter_nodes)

        nodes_list = list(graph.nodes())
        edges_list = list(graph.edges())
        topological_order = list(nx.topological_sort(graph))

        data = {}
        children_list = []

        for edge in edges_list:
            children_list.append(edge[1])

        children_list = list(set(children_list))

        for node in nodes_list:
            data[node] = []
            if node in inter_nodes:
                data[node] = np.random.uniform(low=dict_nodes_thres[node][0], high=dict_nodes_thres[node][0]+1, size=n)
                # for i in range(n):
                #     if (i+1)%max_anomaly == 1:
                #         data[node][i] = dict_nodes_base[node][0]
            else:
                data[node] = np.array([dict_nodes_base[node][0] for i in range(n)])

        # transfer interventions
        for node in topological_order:
            parents = list(graph.predecessors(node))
            if len(parents) != 0:
                for i in range(len(data[node])):
                    # abnormal_parent = False
                    abnormal_parent = 0
                    for par in parents:
                        par_values = data[par]
                        lag = dict_edges_lag[str((par, node))]
                        if i-lag >=0 and par_values[i-lag] >= dict_nodes_thres[par][0]:
                            # abnormal_parent = True
                            abnormal_parent += 1
                    # if abnormal_parent and np.random.uniform() < 1-epsilon:
                    #     data[node][i] = np.random.uniform(low=dict_nodes_thres[node][0], high=1)
                    if self_loops and i-self_lag >=0 and data[node][i-self_lag] >= dict_nodes_thres[node][0]:
                        abnormal_parent +=1
                    if abnormal_parent != 0:
                        impacted = False
                        for m in range(abnormal_parent):
                            if np.random.uniform() < 1-epsilon:
                                impacted = True # data[node][i] += dict_nodes_thres[node][0]
                        if impacted:
                            for par in parents:
                                lag = dict_edges_lag[str((par, node))]
                                if i-lag >=0:
                                    data[node][i]+=dict_edges_coffe[str((par, node))]*data[par][i-lag]
                            if self_loops and i-self_lag >=0:
                                data[node][i]+=dict_edges_coffe[str((node, node))]*data[node][i-self_lag]
                            data[node][i] = max(dict_nodes_thres[node][0], data[node][i])
                    if i-max_anomaly-1>=0 and np.sum(data[node][i-max_anomaly-1:i-1]>=dict_nodes_thres[node][0])==max_anomaly:
                        data[node][i] = dict_nodes_base[node][0]
            else:
                 continue

        data = pd.DataFrame(data)

        # only_inter = True
        # ## *****************************************************************************************************
        # ## Check data
        # print(json_file_path.split('/')[1].split('.')[0])
        # print("****Check impacts of the intervention node****")
        # for node in inter_nodes:
        #     print('Intervention node: ' + node)
        #     print('Descendants of the intervention node:' + str(list(nx.descendants(graph, node))))
        #     check_impact_of_intervention_node(graph=graph, inter_node=node,
        #                                       data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
        # if not only_inter:
        #     print("****Check impacts of the root****")
        #     root_nodes = [i for i in nodes_list if i not in children_list and i not in inter_nodes]
        #     for root in root_nodes:
        #         print('root:' + root)
        #         check_impact_of_intervention_node(graph=graph, inter_node=root,
        #                                   data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
        # ## *****************************************************************************************************

        info = {'nodes_thres': dict_nodes_thres, 'nodes_base': dict_nodes_base, 'edges_lag': dict_edges_lag, 'edges_coffe': dict_edges_coffe, 'intervention_node': list(inter_nodes)}

        data.to_csv(os.path.join(save_data_path, json_file_path.split('/')[1].replace('graph', 'data').replace('json', 'csv')), index=False)

        data_info_path = os.path.join(save_info_path, json_file_path.split('/')[1].replace('graph', 'info'))
        # Save the dictionary as a JSON file
        with open(data_info_path, 'w') as json_file:
            json.dump(info, json_file)


# graph_path: path of the graph file
# intervention_path: path of the intervention node file
# data_path: path to save generated data folder
# info_path: path to save thresholds of nodes and lags of edges folder
# n : number of sampling points
# gamma_min: minimum lag
# gamma_max: maximum lag
# thres_min: minimum threshold for all nodes
#thres_max: maximum threshold for all nodes
# def generate_simulation_data_by_file(graph_path, intervention_path, data_path, info_path, n, gamma_min,
#                              gamma_max, thres_min, thres_max, epsilon, seed=3344, only_inter=False, plot_data=False, visual_graph=False):
#     np.random.seed(seed=seed)
#     # check the existence of data folder
#     if not os.path.exists(data_path):
#         # If it doesn't exist, create the folder
#         os.makedirs(data_path)
#     # check the existence of data information folder
#     if not os.path.exists(info_path):
#         # If it doesn't exist, create the folder
#         os.makedirs(info_path)
#
#     with open(graph_path, 'r') as json_file:
#         json_graph = json.load(json_file)
#
#     inter_file = pd.read_csv(intervention_path)
#     inter_node = inter_file.columns[0]
#
#     # Convert the loaded JSON data into a NetworkX graph
#     graph = dict_to_graph(graph_dict=json_graph, inter_node=inter_node)
#
#     nodes_list = list(graph.nodes())
#     edges_list = list(graph.edges())
#     topological_order = list(nx.topological_sort(graph))
#
#     data = {}
#     dict_nodes_thres = {}
#     children_list = []
#     dict_edges_lag = {}
#
#     for edge in edges_list:
#         dict_edges_lag[edge] = np.random.randint(low=gamma_min, high=gamma_max+1)
#         children_list.append(edge[1])
#
#     children_list = list(set(children_list))
#
#     for node in nodes_list:
#             dict_nodes_thres[node] = [np.round(np.random.uniform(low = thres_min, high = thres_max), 2)]
#
#     if only_inter:
#         for node in nodes_list:
#             data[node] = []
#             if node == inter_node:
#                 data[node] = np.random.uniform(low=0, high=1, size=n)
#             else:
#                 data[node] = np.random.uniform(low=0, high=dict_nodes_thres[node], size=n)
#     else:
#         for node in nodes_list:
#             data[node] = []
#             if node not in children_list or node == inter_node:
#                 data[node] = np.random.uniform(low=0, high=1, size=n)
#             else:
#                 data[node] = np.random.uniform(low=0, high=dict_nodes_thres[node], size=n)
#
#     # transfer interventions
#     for node in topological_order:
#         values = data[node]
#         children = list(graph.successors(node))
#         if len(children) != 0:
#             for i in range(len(values)):
#                 if values[i] >= dict_nodes_thres[node]:
#                     for child in children:
#                         lag = dict_edges_lag[(node,child)]
#                         if i + lag < n and np.random.uniform() < 1-epsilon:
#                             data[child][i+lag] = np.random.uniform(low=dict_nodes_thres[child], high=1)
#         else:
#              continue
#
#     data = pd.DataFrame(data)
#
#     ## *****************************************************************************************************
#     ## Check data
#     print("****Check impacts of the intervention node****")
#     print('Intervention node: ' + inter_node)
#     print('Descendants of the intervention node:' + str(list(nx.descendants(graph, inter_node))))
#     check_impact_of_intervention_node(graph=graph, inter_node=inter_node,
#                                       data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
#     if not only_inter:
#         print("****Check impacts of the root****")
#         root_nodes = [i for i in nodes_list if i not in children_list]
#         for root in root_nodes:
#             print('root:' + root)
#             check_impact_of_intervention_node(graph=graph, inter_node=root,
#                                       data=data, dict_nodes_thres=dict_nodes_thres, dict_edges_lag=dict_edges_lag, n=n)
#     ## *****************************************************************************************************
#
# #     if visual_graph:
# #         visualize_a_graph(graph)
# #     if plot_data:
# #         print(graph_path.split('/')[-1].split('.')[0])
# #         print("****Show the abnormal rate for each variable****")
# #         node_in_order = list(data.keys())
# #         node_in_order.sort()
# #         for node in node_in_order:
# #             print(node + ' abnormal rate : '+ str(np.mean(data[node] >= dict_nodes_thres[node])))
# #         print("****Show the threshold for each variable****")
# #         print(dict_nodes_thres)
# #         print("****Show the lag for each connection****")
# #         print(dict_edges_lag)
# #         plot_data_with_thres(data=data, n=n, dict_nodes_thres=dict_nodes_thres)
#
#     edges_lag = {}
#     for key,value in dict_edges_lag.items():
#         edges_lag[str(key)] = value
#
#     info = {'nodes_thres': dict_nodes_thres, 'edges_lag':edges_lag, 'intervention_node':[node for node in nodes_list if node not in children_list]}
#
#
#     data.to_csv(os.path.join(data_path, graph_path.split('/')[1].replace('graph', 'data').replace('json', 'csv')), index=False)
#
#     data_info_path = os.path.join(info_path, graph_path.split('/')[1].replace('graph', 'info'))
#     # Save the dictionary as a JSON file
#     with open(data_info_path, 'w') as json_file:
#         json.dump(info, json_file)

if __name__ =='__main__':
    # # Generate historical data
    # graphs_path = 'graphs'
    # n = 20000 # sampling points
    # gamma_min = 1
    # gamma_max = 1
    # thres_min = 0.7
    # thres_max = 0.9
    # epsilon = 0.3
    # prob_inter = 0.1
    # self_loops = True
    # max_anomaly = 5
    # # data_path = os.path.join('E1', 'certain_SL', 'historical_data_20000')
    # # save_info_path = os.path.join('E1','certain_SL', 'data_info_20000')
    # # info_path = os.path.join('E1','certain_SL', 'data_info')
    # data_path = os.path.join('E1', 'uncertain_' + str(epsilon) + '_SL', 'historical_data_20000')
    # save_info_path = os.path.join('E1', 'uncertain_' + str(epsilon) + '_SL', 'data_info_20000')
    # info_path = os.path.join('E1', 'uncertain_' + str(epsilon) + '_SL', 'data_info')
    # # generate_historical_data_by_folder_PC(graphs_path=graphs_path, data_path=data_path, info_path=info_path, n=n,
    # #                                    gamma_min=gamma_min, gamma_max=gamma_max, thres_min=thres_min,
    # #                                    thres_max=thres_max, prob_inter=prob_inter, epsilon=epsilon,
    # #                                    self_loops=self_loops, max_anomaly=max_anomaly)
    # extend_historical_data_by_folder_PC(graphs_path=graphs_path, data_path=data_path, info_path=info_path,
    #                                     save_info_path=save_info_path, n=n, gamma_min=gamma_min, gamma_max=gamma_max,
    #                                     thres_min=thres_min, thres_max=thres_max, prob_inter=prob_inter, epsilon=epsilon,
    #                                     self_loops=self_loops, max_anomaly=max_anomaly)


    # # example: generate simulation data for the whole folder PC
    # num_inters = 2
    # max_anomaly = 5
    #
    # sampling_number = [2000]
    # scenarios = ['certain', 'uncertain']
    # whether_self_loops = [True, False]
    # whether_in_same_path = [True, False]
    # for n in sampling_number:
    #     for certainty in scenarios:
    #         for self_loops in whether_self_loops:
    #             for in_same_path in whether_in_same_path:
    #                 folder_name = certainty
    #                 if certainty == 'certain':
    #                     epsilon = 0
    #                 else:
    #                     epsilon = 0.3
    #                     folder_name += '_'+str(epsilon)
    #                 if self_loops:
    #                     folder_name += '_SL'
    #                 if in_same_path:
    #                     data_file_name = 'actual_data_same_path_'+str(num_inters)+'_inters_'+str(n)
    #                     info_file_name = 'data_info_same_path_'+str(num_inters)+'_inters_'+str(n)
    #                 else:
    #                     data_file_name = 'actual_data_'+str(num_inters)+'_inters_'+str(n)
    #                     info_file_name = 'data_info_'+str(num_inters)+'_inters_'+str(n)
    #
    #                 graphs_path = 'graphs'
    #                 info_path = os.path.join('PC', folder_name, 'data_info')
    #                 save_data_path = os.path.join('PC', folder_name, data_file_name)
    #                 save_info_path = os.path.join('PC', folder_name, info_file_name)
    #                 generate_simulation_data_by_folder_PC(graphs_path=graphs_path, info_path=info_path, save_data_path=save_data_path, save_info_path=save_info_path,
    #                                    n=n, epsilon=epsilon, only_inter=True, num_inters=num_inters, self_loops=self_loops, max_anomaly=max_anomaly, in_same_path=in_same_path)


    # Generate historical data SK
    # graphs_path = 'graphs'
    # n = 20000 # sampling points
    # gamma_min = 1
    # gamma_max = 1
    # thres_min = 0.7
    # thres_max = 0.9
    # epsilon = 0.3
    # prob_inter = 0.1
    # self_loops = True
    # max_anomaly = 5
    # # data_path = os.path.join('E2', 'certain_SL', 'historical_data_20000')
    # # save_info_path = os.path.join('E2', 'certain_SL', 'data_info_20000')
    # # info_path = os.path.join('E2', 'certain_SL', 'data_info')
    # data_path = os.path.join('E2', 'uncertain_' + str(epsilon)+ '_SL', 'historical_data_20000')
    # save_info_path = os.path.join('E2', 'uncertain_' + str(epsilon)+ '_SL', 'data_info_20000')
    # info_path = os.path.join('E2', 'uncertain_' + str(epsilon)+ '_SL', 'data_info')
    # # generate_historical_data_by_folder_SK(graphs_path=graphs_path, save_data_path=data_path, save_info_path=info_path, n=n,
    # #                                    gamma_min=gamma_min, gamma_max=gamma_max, thres_min=thres_min,
    # #                                    thres_max=thres_max, prob_inter=prob_inter, epsilon=epsilon,
    # #                                    self_loops=self_loops, max_anomaly=max_anomaly)
    # extend_historical_data_by_folder_SK(graphs_path=graphs_path, save_data_path=data_path, info_path=info_path,
    #                                     save_info_path=save_info_path, n=n, gamma_min=gamma_min, gamma_max=gamma_max,
    #                                     thres_min=thres_min, thres_max=thres_max, prob_inter=prob_inter, epsilon=epsilon,
    #                                    self_loops=self_loops, max_anomaly=max_anomaly)



    # generate simulated data SK
    # num_inters = 2
    # max_anomaly = 5
    #
    # sampling_number = [2000]
    # scenarios = ['certain', 'uncertain']
    # whether_self_loops = [True, False]
    # whether_in_same_path = [True, False]
    # for n in sampling_number:
    #     for certainty in scenarios:
    #         for self_loops in whether_self_loops:
    #             for in_same_path in whether_in_same_path:
    #                 folder_name = certainty
    #                 if certainty == 'certain':
    #                     epsilon = 0
    #                 else:
    #                     epsilon = 0.3
    #                     folder_name += '_'+str(epsilon)
    #                 if self_loops:
    #                     folder_name += '_SL'
    #                 if in_same_path:
    #                     data_file_name = 'actual_data_same_path_'+str(num_inters)+'_inters_'+str(n)
    #                     info_file_name = 'data_info_same_path_'+str(num_inters)+'_inters_'+str(n)
    #                 else:
    #                     data_file_name = 'actual_data_'+str(num_inters)+'_inters_'+str(n)
    #                     info_file_name = 'data_info_'+str(num_inters)+'_inters_'+str(n)
    #
    #                 graphs_path = 'graphs'
    #                 info_path = os.path.join('SK', folder_name, 'data_info')
    #                 # save_data_path = os.path.join('certain_', 'data')
    #                 #save_info_path = os.path.join('certain_', 'data_info')
    #                 # save_data_path = os.path.join('SK', 'certain_SL', 'actual_data_same_path_'+str(num_inters)+'_inters_'+str(n))
    #                 # save_info_path = os.path.join('SK', 'certain_SL', 'data_info_same_path_'+str(num_inters)+'_inters_'+str(n))
    #                 save_data_path = os.path.join('SK', folder_name, data_file_name)
    #                 save_info_path = os.path.join('SK', folder_name, info_file_name)
    #                 generate_simulation_data_by_folder_SK(graphs_path=graphs_path, info_path=info_path, save_data_path=save_data_path, save_info_path=save_info_path,
    #                                                       n=n, num_inters=num_inters, epsilon=epsilon, self_loops=self_loops, max_anomaly=max_anomaly, in_same_path=in_same_path)



    ## example: generate data according to the file of the graph
    # graph_path = 'graphs/graph_1_2_0.json'
    # intervention_path = 'interventions/intervention_1_2_0.csv'
    # data_path = 'data'
    # info_path = 'data_info'
    # n = 500 # sampling points
    # gamma_min = 1
    # gamma_max = 3
    # thres_min = 0.7
    # thres_max = 0.9
    #
    # generate_simulation_data_by_file(graph_path=graph_path, intervention_path=intervention_path, data_path=data_path, info_path=info_path, n=n,
    #                      gamma_min=gamma_min, gamma_max=gamma_max, thres_min=thres_min, thres_max=thres_max, seed=3344, only_inter=False)
