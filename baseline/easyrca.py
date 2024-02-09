"""
Coded by Charles Assaad, Simon Ferreira, Imad Ez-Zejjari, and Lei Zan
"""

import pandas as pd
import numpy as np
from baseline.estimation import grubb_test, LinearRegression
from baseline.identification.identifiability_of_direct_effect_in_scg import *


class NotIdentifiableError(Exception):
    pass


class EasyRCA:
    def __init__(self, summary_graph, anomalous_nodes, anomalies_start_time=None, anomaly_length=200, gamma_max=1,
                 sig_threshold=0.05, acyclic_adjustment_set="ParentsY", adjustment_set="AncestorsY",
                 differentiate_structural_and_parametric=True):
        # TODO add a parameter distinguish_anomaly_type which chooses between one of the strategies for when to
        #  distinguish between structural and parametric:
        # always, except_subroots, except_time_defying, except_subroots_and_time_defying, never.
        # In some cases (probably all but "never"), every direct effect should be estimated so that an intervention
        # which is structural *and* parametric is counted as structural.
        # This might be implemented as a level of confidence of each type of root cause (subroot, time defying,
        # structural, parametric) so that EasyRCA tries to find the ones with the most confidence first.
        """
        :param summary_graph: networkx graph
        :param anomalous_nodes: list
        :param anomalies_start_time: dict
        :param anomaly_length: int
        :param gamma_max: int
        :param sig_threshold: float
        :param acyclic_adjustment_set: str
        :param adjustment_set: str
        :param differentiate_structural_and_parametric: bool
        """
        self.summary_graph = summary_graph
        self.scg_no_self_loops = remove_self_loops(summary_graph)
        self.anomalous_nodes = anomalous_nodes
        self.anomalies_start_time = anomalies_start_time
        self.anomaly_length = anomaly_length
        self.gamma_max = gamma_max
        self.sig_threshold = sig_threshold
        self.acyclic_adjustment_set = acyclic_adjustment_set
        self.adjustment_set = adjustment_set
        self.adjustment_sets = {"ParentsY": adjustment_set_for_direct_effect_in_ascgl_using_ParentsY,
                                "ParentsXY": adjustment_set_for_direct_effect_in_ascgl_using_ParentsXY,
                                "AncestorsY": smaller_adjustment_set_for_direct_effect_in_scg,
                                "All": huge_adjustment_set_for_direct_effect_in_scg}
        self.differentiate_structural_and_parametric = differentiate_structural_and_parametric
        self.dict_linked_anomalous_graph = dict()
        self._find_linked_anomalous_graphs()

        self.root_causes = dict()
        for id_linked_anomalous_graph in self.dict_linked_anomalous_graph.keys():
            self.root_causes[id_linked_anomalous_graph] = {"roots": set(), "time_defying": set(), "data_defying": set()}
        if self.differentiate_structural_and_parametric:
            # todo differentiate also structural and parametric for roots and time_defying
            self.data_defying_root_causes = dict()
            for id_linked_anomalous_graph in self.dict_linked_anomalous_graph.keys():
                self.data_defying_root_causes[id_linked_anomalous_graph] = {"structural": set(), "parametric": set()}

        self.get_recommendations = pd.DataFrame()

        # indicates if data was used or not
        self.search_rc_from_graph = False
        # indicates if data was used or not
        self.search_rc_from_data = False

        # Minimum lag between each edge in the graph: if both nodes in an edges are anomalous then detect the min lag by
        # looking at the time of the appearance of anomalies, if one the nodes is not anomalous then min lag is 0
        self.gamma_min_dict = dict()
        self.d_sep_by_empty_in_manip_graph = dict()
        self._get_gamma_min()

        self.nodes_to_temporal_nodes = dict()
        for node in self.summary_graph.nodes:
            temporal_node = str(node) + "_t"
            self.nodes_to_temporal_nodes[node] = [temporal_node] + [temporal_node + "-" + str(gamma) for gamma in
                                                                    range(1, 2 * self.gamma_max + 1)]

    def _get_gamma_min(self):
        """
        Find Minimum lag between each edge in the graph: if both nodes in an edges are anomalous then detect the min lag
        by looking at the time of the appearance of anomalies, if one the nodes is not anomalous then min lag is 0
        """
        for edge in self.summary_graph.edges:
            if edge[0] == edge[1]:
                self.gamma_min_dict[edge] = 1
            elif (edge[0] in self.anomalous_nodes) and  (edge[1] in self.anomalous_nodes):
                self.gamma_min_dict[edge] = max(0, self.anomalies_start_time[edge[1]] -
                                                self.anomalies_start_time[edge[0]])
            else:
                self.gamma_min_dict[edge] = 0

    def _find_linked_anomalous_graphs(self):
        """
        Find linked anomalous graphs, given the initial summary causal graph by looking grouping all anomalous
        nodes that have an undirected path between them
        """
        undirected_anomalous_graph = self.summary_graph.subgraph(self.anomalous_nodes).copy().to_undirected()
        id_linked_anomalous_graph = 0
        for linked_anomalous_nodes in nx.connected_components(undirected_anomalous_graph):
            linked_anomalous_graph = self.summary_graph.subgraph(linked_anomalous_nodes).copy()
            self.dict_linked_anomalous_graph[id_linked_anomalous_graph] = linked_anomalous_graph
            id_linked_anomalous_graph += 1

    def return_other_possible_root_causes(self, id_linked_anomalous_graph, not_allowed_vertices=[]):
        """
        Save all nodes that are potentially root causes but were not detected root nodes nor time defying nodes
        :param id_linked_anomalous_graph: the id of linked anomalous graph
        :param not_allowed_vertices: list (optional)
        """
        return set(self.dict_linked_anomalous_graph[id_linked_anomalous_graph].nodes).difference(
            self.root_causes[id_linked_anomalous_graph]["roots"],
            self.root_causes[id_linked_anomalous_graph]["time_defying"],
            self.root_causes[id_linked_anomalous_graph]["data_defying"], not_allowed_vertices)

    def _search_roots(self, id_linked_anomalous_graph):
        """
        Find roots of a given linked anomalous graph
        :param id_linked_anomalous_graph: the id of linked anomalous graph
        :return: Void
        """
        linked_anomalous_graph = self.dict_linked_anomalous_graph[id_linked_anomalous_graph]
        linked_anomalous_graph_no_self_loops = remove_self_loops(linked_anomalous_graph)
        possible_root_causes = self.return_other_possible_root_causes(id_linked_anomalous_graph)
        roots = {node for node in possible_root_causes if
                 not linked_anomalous_graph_no_self_loops.in_degree(node)}
        self.root_causes[id_linked_anomalous_graph]["roots"].update(roots)

    def _search_time_defiance(self, id_linked_anomalous_graph):
        """
        Use time about the first appearance of anomalies to find nodes that temporally defy the causal structure
        :param id_linked_anomalous_graph: the id of linked anomalous graph
        :return: Void
        """
        linked_anomalous_graph = self.dict_linked_anomalous_graph[id_linked_anomalous_graph]
        linked_anomalous_graph_no_self_loops = remove_self_loops(linked_anomalous_graph)
        possible_root_causes = self.return_other_possible_root_causes(id_linked_anomalous_graph)

        def exists_parents_with_old_anomaly_appearance(v):
            for p in linked_anomalous_graph_no_self_loops.predecessors(v):
                if self.anomalies_start_time[p] <= self.anomalies_start_time[v]:
                    return True
            return False
        time_defying = {node for node in possible_root_causes if not exists_parents_with_old_anomaly_appearance(node)}
        self.root_causes[id_linked_anomalous_graph]["time_defying"].update(time_defying)

    def _process_data(self, data):
        new_data = pd.DataFrame()
        for gamma in range(0, 2 * self.gamma_max + 1):
            shifteddata = data.shift(periods=-2 * self.gamma_max + gamma)

            new_columns = []
            for node in data.columns:
                new_columns.append(self.nodes_to_temporal_nodes[node][gamma])
            shifteddata.columns = new_columns
            new_data = pd.concat([new_data, shifteddata], axis=1, join="outer")
        new_data.dropna(axis=0, inplace=True)

        # divide new data into two: normal and anomalous
        last_start_time_normal = 0
        first_end_time_normal = self.anomalies_start_time[self.anomalous_nodes[0]] - 1
        for node in self.anomalous_nodes:
            first_end_time_normal = min(first_end_time_normal, self.anomalies_start_time[node] - 1)
        normal_data = new_data.loc[last_start_time_normal:first_end_time_normal]

        last_start_time_anomaly = 0
        for node in self.anomalous_nodes:
            last_start_time_anomaly = max(last_start_time_anomaly, self.anomalies_start_time[node])
        first_end_time_anomaly = last_start_time_anomaly + self.anomaly_length
        anomalous_data = new_data.loc[last_start_time_anomaly:first_end_time_anomaly]

        return normal_data, anomalous_data

    def _search_data_defiance(self, id_linked_anomalous_graph, normal_data, anomalous_data):
        """
        sds
        :param id_linked_anomalous_graph: the id of linked anomalous graph
        :param normal_data: Dataframe
        :param anomalous_data: Dataframe
        :return:
        """
        batch_size = anomalous_data.shape[0]
        split_nb = int(normal_data.shape[0]/batch_size)
        # # ################################"
        # if split_nb == 0:
        #     split_nb = 1
        # # ################################
        normal_data_batchs = np.array_split(normal_data, split_nb)
        linked_anomalous_graph = self.dict_linked_anomalous_graph[id_linked_anomalous_graph]
        linked_anomalous_graph_no_self_loops = remove_self_loops(linked_anomalous_graph)
        if self.acyclic_adjustment_set and is_ascgl(linked_anomalous_graph):
            adjustment_set = self.acyclic_adjustment_set
        else:
            adjustment_set = self.adjustment_set
        possible_root_causes = self.return_other_possible_root_causes(id_linked_anomalous_graph)
        for edge in linked_anomalous_graph_no_self_loops.edges:
            x = edge[0]
            y = edge[1]
            if y in possible_root_causes:
                cond_dict = dict()
                for gamma_xy in range(self.gamma_min_dict[edge], self.gamma_max + 1):
                    if not is_identifiable(linked_anomalous_graph, x, y, gamma_xy, self.gamma_max):
                        cond_dict[gamma_xy] = None
                    else:
                        direct_effect, cond_set = self.adjustment_sets[adjustment_set](linked_anomalous_graph, x, y,
                                                                                       gamma_xy, self.gamma_max,
                                                                                       self.gamma_min_dict)
                        if direct_effect:
                            raise ValueError("This should not happen as (x,y) is an edge and gamma_xy <= gamma_max")
                        else:
                            cond_dict[gamma_xy] = cond_set
                    # for gamma_xy in sorted(cond_dict.keys(), reverse = True):
                    cond_set = cond_dict[gamma_xy]
                    yt = self.nodes_to_temporal_nodes[y][0]
                    xt = self.nodes_to_temporal_nodes[x][gamma_xy]
                    if cond_set is not None:
                        cond_list = [self.nodes_to_temporal_nodes[v][gamma] for (v, gamma) in cond_set]
                        ci = LinearRegression(xt, yt, cond_list)
                        pval_normal = ci.test_zeo_coef(normal_data)
                        if pval_normal < self.sig_threshold:
                            coeff_anomalous = ci.get_coeff(anomalous_data)
                            pval_list = list()
                            pval_list.append(coeff_anomalous)
                            for i in range(split_nb - 1):
                                coeff_normal = ci.get_coeff(normal_data_batchs[i])
                                pval_list.append(coeff_normal)
                            grubb_res = grubb_test(pval_list, confidence_level=self.sig_threshold)
                            if grubb_res["anomaly_position"] == 0:
                                self.root_causes[id_linked_anomalous_graph]["data_defying"].add(y)
                                if self.differentiate_structural_and_parametric:
                                    if ci.test_zeo_coef(anomalous_data) >= self.sig_threshold:
                                        self.data_defying_root_causes[id_linked_anomalous_graph]["structural"].add(y)
                                    else:
                                        self.data_defying_root_causes[id_linked_anomalous_graph]["parametric"].add(y)
                    else:
                        # todo potential parents
                        raise NotIdentifiableError("direct effect from " + xt + " to " + yt + " is not identifiable")

    @staticmethod
    def _sorted_by_reach(rc, linked_anomalous_graph_no_self_loops):
        """
        :param rc: list
        :param linked_anomalous_graph_no_self_loops: Networkx graph without loops
        :return: list of nodes ordered by reach heuristic
        """
        rc_list = list(rc)
        list_nb_descendants = []
        for node in rc_list:
            list_nb_descendants.append(len(nx.descendants(linked_anomalous_graph_no_self_loops, node)))
        sorted_id = sorted(range(len(list_nb_descendants)), key=list_nb_descendants.__getitem__, reverse=True)
        new_rc_list = []
        count_iter = 0
        for i in range(len(sorted_id)):
            if i == 0:
                new_rc_list.append([rc_list[sorted_id[i]]])
                count_iter = count_iter + 1
            else:
                if list_nb_descendants[sorted_id[i]] == list_nb_descendants[sorted_id[i-1]]:
                    new_rc_list[count_iter - 1].append(rc_list[sorted_id[i]])
                else:
                    new_rc_list.append([rc_list[sorted_id[i]]])
                    count_iter = count_iter + 1
        return new_rc_list

    @staticmethod
    def _sorted_by_reach_for_sets(rc_list_of_list, linked_anomalous_graph_no_self_loops):
        """
        :param rc_list_of_list: list of lists
        :param linked_anomalous_graph_no_self_loops: Networkx graph without loops
        :return: list of sets ordered by reach heuristic
        """
        list_nb_descendants = []
        rc_list_of_list = list(rc_list_of_list)
        for rc_list in rc_list_of_list:
            set_descendants = set()
            for node in rc_list:
                set_descendants.update(set(nx.descendants(linked_anomalous_graph_no_self_loops, node)))
            list_nb_descendants.append(len(set_descendants))
        sorted_id = sorted(range(len(list_nb_descendants)), key=list_nb_descendants.__getitem__, reverse=True)
        new_rc_list_of_list = []
        count_iter = 0
        for i in range(len(sorted_id)):
            if i == 0:
                new_rc_list_of_list.append([rc_list_of_list[sorted_id[i]]])
                count_iter = count_iter + 1
            else:
                if list_nb_descendants[sorted_id[i]] == list_nb_descendants[sorted_id[i-1]]:
                    new_rc_list_of_list[count_iter - 1].append(rc_list_of_list[sorted_id[i]])
                else:
                    new_rc_list_of_list.append([rc_list_of_list[sorted_id[i]]])
                    count_iter = count_iter + 1
        return new_rc_list_of_list

    def action_recommendation(self, confidence_level=True):
        """
        Recommend actions to the user
        :param confidence_level: If True return a confidence level such that 1: True Root cause(s), 2: The set contain
                                                                at least one true root cause, 3: Potential root cause(s)
        :return: Dataframe which contains the ranking of actions that needs to be done on root causes for each linked
                                                                                                        anomalous graph
        """
        self.get_recommendations = pd.DataFrame()
        set_containing_at_least_one_unknown_root_cause = dict()
        for id_linked_anomalous_graph in self.dict_linked_anomalous_graph.keys():
            set_containing_at_least_one_unknown_root_cause[id_linked_anomalous_graph] = []

        for id_linked_anomalous_graph in self.dict_linked_anomalous_graph.keys():
            linked_anomalous_graph = self.dict_linked_anomalous_graph[id_linked_anomalous_graph]
            linked_anomalous_graph_no_self_loops = remove_self_loops(linked_anomalous_graph)
            sorted_list = []

            if self.search_rc_from_graph:
                rc_from_graph = ["roots", "time_defying"]
                for rc_name in rc_from_graph:
                    rc = self.root_causes[id_linked_anomalous_graph][rc_name]
                    rc_sorted_by_reach = self._sorted_by_reach(rc, linked_anomalous_graph_no_self_loops)
                    if confidence_level:
                        rc_sorted_by_reach = [(x, 1) for x in rc_sorted_by_reach]
                    sorted_list = sorted_list + rc_sorted_by_reach

            if self.search_rc_from_data:
                if self.differentiate_structural_and_parametric:
                    rc_from_data = ["structural", "parametric"]
                    for rc_name in rc_from_data:
                        rc = self.data_defying_root_causes[id_linked_anomalous_graph][rc_name]
                        rc_sorted_by_reach = self._sorted_by_reach(rc, linked_anomalous_graph_no_self_loops)
                        if confidence_level:
                            rc_sorted_by_reach = [(x, 1) for x in rc_sorted_by_reach]
                        sorted_list = sorted_list + rc_sorted_by_reach
                else:
                    rc = self.root_causes[id_linked_anomalous_graph]["data_defying"]
                    rc_sorted_by_reach = self._sorted_by_reach(rc, linked_anomalous_graph_no_self_loops)
                    if confidence_level:
                        rc_sorted_by_reach = [(x, 1) for x in rc_sorted_by_reach]
                    sorted_list = sorted_list + rc_sorted_by_reach
            else:
                # Construct other possible root causes
                if len(self.root_causes[id_linked_anomalous_graph]["roots"]) == 0:
                    cycles = nx.recursive_simple_cycles(linked_anomalous_graph_no_self_loops)
                    for cycle in cycles:
                        test_if_subroot_cycle = all([set(linked_anomalous_graph_no_self_loops.predecessors(c)).issubset(
                                                         cycle) for c in cycle])
                        if test_if_subroot_cycle:
                            test_if_no_time_defying_in_cycle = all(
                                [c not in self.root_causes[id_linked_anomalous_graph]["time_defying"] for c in cycle])
                            print(test_if_no_time_defying_in_cycle, "11")
                            if test_if_no_time_defying_in_cycle:
                                set_containing_at_least_one_unknown_root_cause[id_linked_anomalous_graph].append(cycle)
                    prc_sorted_by_reach = self._sorted_by_reach_for_sets(set_containing_at_least_one_unknown_root_cause[
                                                                             id_linked_anomalous_graph],
                                                                         linked_anomalous_graph_no_self_loops)
                    if confidence_level:
                        prc_sorted_by_reach = [(x, 2) for x in prc_sorted_by_reach]
                    sorted_list = sorted_list + prc_sorted_by_reach

                not_allowed_vertices = [item for sublist in
                                        set_containing_at_least_one_unknown_root_cause[id_linked_anomalous_graph]
                                        for item in sublist]
                prc = self.return_other_possible_root_causes(id_linked_anomalous_graph, not_allowed_vertices)
                prc_sorted_by_reach = self._sorted_by_reach(prc, linked_anomalous_graph_no_self_loops)
                if confidence_level:
                    prc_sorted_by_reach = [(x, 3) for x in prc_sorted_by_reach]
                sorted_list = sorted_list + prc_sorted_by_reach

            if len(self.get_recommendations.columns) == 0:
                self.get_recommendations["LinkedAnomalousGraph_" + str(id_linked_anomalous_graph)] = pd.Series(sorted_list)
            else:
                self.get_recommendations = pd.concat([self.get_recommendations,
                                                      pd.Series(sorted_list,
                                                                name="LinkedAnomalousGraph_" +
                                                                     str(id_linked_anomalous_graph)).to_frame()],
                                                     axis=1, join="outer")
        self.get_recommendations.dropna(axis=0, how='all', inplace=True)

    def run_without_data(self):
        self.search_rc_from_graph = True
        for id_linked_anomalous_graph in self.dict_linked_anomalous_graph.keys():
            self._search_roots(id_linked_anomalous_graph)
            self._search_time_defiance(id_linked_anomalous_graph)

    def run(self, data):
        self.search_rc_from_data = True
        normal_data, anomalous_data = self._process_data(data)
        for id_linked_anomalous_graph in self.dict_linked_anomalous_graph.keys():
            if not self.search_rc_from_graph:
                self._search_roots(id_linked_anomalous_graph)
                self._search_time_defiance(id_linked_anomalous_graph)
            self._search_data_defiance(id_linked_anomalous_graph, normal_data, anomalous_data)


if __name__ == '__main__':
    type_of_graph = "acyclic"
    np.random.seed(1)
    graph = nx.DiGraph()
    if type_of_graph == "acyclic":
        # graph.add_edges_from([("y", "z"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"), ("d", "f"), ("e", "f")])
        graph.add_edges_from([("y", "z"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"), ("c", "f"), ("b", "f"),
                              ("d", "e")])

    else:
        # graph.add_edges_from([("y", "z"), ("a", "b"), ("c", "a"), ("b", "c"), ("b", "d"), ("c", "e"), ("d", "f"),
        #                       ("e", "f")])
        graph.add_edges_from([("y", "z"), ("a", "b"), ("c", "a"), ("b", "c"), ("b", "d"), ("c", "e"), ("d", "e"),
                              ("e", "d"),  ("d", "f"), ("e", "f")])
        # graph.add_edges_from([("y", "z"), ("z", "y"), ("z", "f"), ("a", "b"), ("c", "a"), ("b", "c"), ("b", "d"),
        #                       ("c", "e"), ("d", "e"), ("e", "d"), ("d", "f"),("e", "f")])
        # graph.add_edges_from([("y", "k"), ("y", "l"), ("y", "m")])
    graph.add_edges_from([("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("e", "e"), ("f", "f"), ("y", "z"),
                          ("z", "z")])
    anomalous = ["a", "b", "c", "e", "f", "z", "y", "d"]

    # anomalous = ["a", "b", "c", "e", "f", "z", "y", "d", "k", "l", "m"]
    anomalies_start = dict()
    anomalies_start["a"] = 20000
    anomalies_start["y"] = 20000
    anomalies_start["b"] = 20000
    anomalies_start["c"] = 20000
    anomalies_start["d"] = 20000
    anomalies_start["e"] = 20000
    anomalies_start["f"] = 20000
    anomalies_start["z"] = 19999
    # anomalies_start["b"] = 20001
    # anomalies_start["c"] = 20001
    # anomalies_start["d"] = 20002
    # anomalies_start["e"] = 20002
    # anomalies_start["f"] = 20002
    # anomalies_start["z"] = 19999

    anomaly_size = 200

    draw_graph(graph)
    # find some root causes using only the graph
    AG = EasyRCA(graph, list(graph.nodes), anomalies_start_time=anomalies_start, anomaly_length=anomaly_size,
                 gamma_max=3, sig_threshold=0.01, acyclic_adjustment_set="ParentsY", adjustment_set="AncestorsY")
    AG.run_without_data()
    print(AG.root_causes)
    print(AG.data_defying_root_causes)
    AG.action_recommendation()
    print(AG.get_recommendations)

    # Simulate data
    print("######################################")
    print("Generating data: Start")
    import numpy as np
    np.random.seed(1)
    data_size = 30000

    anomalousTopo = ["a", "y", "b", "c", "d", "e", "f", "z"]
    non_root_anomalousTopo = ["b", "c", "d", "f", "z"]

    coef_dict = {('a', 'b'): 0.47531980423231657, ('a', 'c'): 0.7482920440979423, ('b', 'd'): 0.3001029373356104,
     ('b', 'f'): 0.3720993153686558, ('c', 'e'): 0.23208030173540176, ('c', 'f'): 0.183104735291918,
     ('d', 'e'): 0.2676341902399038, ('a', 'a'): 0.411004654338743, ('b', 'b'): 0.457090726807603,
     ('c', 'c'): 0.5849350606030213, ('d', 'd'): 0.4772750629629653, ('f', 'f'): 0.7166975503570835,
     ('e', 'e'): 0.2840070247583657, ('y', 'z'): 0.3, ('y', 'y'): 1, ('z', 'z'): 0.1}

    noise = pd.DataFrame(np.zeros([data_size, len(graph.nodes)]), columns=anomalous)
    param_data = pd.DataFrame(np.zeros([data_size, len(graph.nodes)]), columns=anomalous)

    for anomalous_node in anomalous:
        param_data[anomalous_node] = 0.1 * np.random.normal(size=data_size)
        noise[anomalous_node] = 0.1 * np.random.normal(size=data_size)

    if type_of_graph == "acyclic":
        for t in range(1, data_size):
            param_data["a"].loc[t] = (coef_dict[("a", "a")] * param_data["a"].loc[t-1] + noise["a"].loc[t])
            param_data["y"].loc[t] = (coef_dict[("y", "y")] * param_data["y"].loc[t-1] + noise["y"].loc[t])
            param_data["b"].loc[t] = (coef_dict[("b", "b")] * param_data["b"].loc[t-1] +
                                      coef_dict[("a", "b")] * param_data["a"].loc[t] + noise["b"].loc[t])
            param_data["c"].loc[t] = (coef_dict[("c", "c")] * param_data["c"].loc[t-1] +
                                      coef_dict[("a", "c")] * param_data["a"].loc[t] + noise["c"].loc[t])
            param_data["d"].loc[t] = (coef_dict[("d", "d")] * param_data["d"].loc[t-1] +
                                      coef_dict[("b", "d")] * param_data["b"].loc[t] + noise["d"].loc[t])
            param_data["e"].loc[t] = (coef_dict[("e", "e")] * param_data["e"].loc[t-1] +
                                      coef_dict[("d", "e")] * param_data["d"].loc[t] +
                                      coef_dict[("c", "e")] * param_data["c"].loc[t] + noise["e"].loc[t])
            param_data["f"].loc[t] = (coef_dict[("f", "f")] * param_data["f"].loc[t-1] +
                                      coef_dict[("b", "f")] * param_data["b"].loc[t] + noise["f"].loc[t])
            param_data["z"].loc[t] = (coef_dict[("z", "z")] * param_data["z"].loc[t-1] +
                                      coef_dict[("y", "z")] * param_data["y"].loc[t] + noise["z"].loc[t])

        print("Intervening on a, y, z and e ...")
        # intervention on root a
        param_data["a"].loc[anomalies_start["a"]: anomalies_start["a"] + anomaly_size - 1] = (
                np.random.normal(1, 1, size=anomaly_size))
        # intervention on root y
        param_data["y"].loc[anomalies_start["y"]: anomalies_start["y"] + anomaly_size - 1] = (
                np.random.normal(1, 1, size=anomaly_size))
        # intervention on e
        param_data["e"].loc[anomalies_start["e"]: anomalies_start["e"] + anomaly_size - 1] = np.random.normal(1, 1, size=anomaly_size)
        # propagate interventions to other vertices
        for t in range(anomalies_start["b"], anomalies_start["b"] + anomaly_size - 1):
            param_data["b"].loc[t] = (coef_dict[("b", "b")] * param_data["b"].loc[t-1] +
                                      coef_dict[("a", "b")] * param_data["a"].loc[t] + noise["b"].loc[t])
            param_data["c"].loc[t] = (coef_dict[("c", "c")] * param_data["c"].loc[t-1] +
                                      coef_dict[("a", "c")] * param_data["a"].loc[t] + noise["c"].loc[t])
            param_data["d"].loc[t] = (coef_dict[("d", "d")] * param_data["d"].loc[t-1] +
                                      coef_dict[("b", "d")] * param_data["b"].loc[t] + noise["d"].loc[t])
            param_data["f"].loc[t] = (coef_dict[("f", "f")] * param_data["f"].loc[t-1] +
                                      coef_dict[("b", "f")] * param_data["b"].loc[t] + noise["f"].loc[t])
            param_data["z"].loc[t] = (coef_dict[("z", "z")] * param_data["z"].loc[t-1] +
                                      coef_dict[("y", "z")] * param_data["y"].loc[t] + noise["z"].loc[t])
    print("Generating data: Done")
    print("######################################")

    # find all root causes using graph and data
    AG.run(param_data)
    print(AG.root_causes)
    print(AG.data_defying_root_causes)
    AG.action_recommendation()
    print(AG.get_recommendations)
