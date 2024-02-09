"""
Coded by Simon Ferreira and Charles Assaad
"""

import networkx as nx
from .tools import *

# -----------------Code of paper {identifiability of direct from summary causal graphs}------------------------

def is_identifiable(sg, x, y, gamma_xy, gamma_max = 1):
    """
    Determines whether the direct effect from x to y with lag gamma_xy is identifiable
    for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
    This corresponds to Theorem 1 in {identifiability of direct from summary causal graphs}.
    :param sg: networkx directed graph
    :param gamma_max: int
    :param x: node
    :param y: node
    :param gamma_xy: int
    :return: bool
    """
    assert x != y
    assert (x in sg.nodes) and (y in sg.nodes)
    assert 0 <= gamma_xy

# This code has exactly the same structure as Theorem 1 in {identifiability of direct effect from summary causal graphs}
    if (not sg.has_edge(x, y)) or (gamma_xy > gamma_max):
        return True

    undirected_sg = sg.to_undirected()
    all_simple_paths = list(nx.all_simple_paths(undirected_sg, x, y))
    if (y, x) not in sg.edges:
        all_simple_paths.remove([x, y])
    for path in all_simple_paths:
        if is_active(sg, path) and is_subset(path[1:-1], nx.descendants(sg, y)):
            if gamma_xy == 0 and is_non_direct(sg, path):
                return False
            if gamma_xy > 0:
                n = len(path)
                if n > 2:
                    path_has_left_arrow = False
                    for i in range(n - 1):
                        if (not sg.has_edge(path[i], path[i + 1])) and sg.has_edge(path[i + 1], path[i]):
                            path_has_left_arrow = True
                            break
                    if not path_has_left_arrow:
                        return False
                if n == 2:
                    if (x in nx.descendants(sg, y)) and (exists_cycle(sg, x, [y])):
                        return False
    return True


def huge_adjustment_set_for_direct_effect_in_scg(sg, x, y, gamma_xy, gamma_max = 1, gamma_min_dict = None):
    #todo use gamma_min_dict
    """
    Gives a single-door set to identify the direct effect from x to y with lag gamma_xy if it is identifiable
    for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
    Otherwise fails.
    This corresponds to Corollary 1 in {identifiability of direct from summary causal graphs}.
    The returned value of this function (x,S) can be interpreted as follows:
    if x != None:
        x is the direct effect.
    else x == None:
        The direct effect is can be estimated using the adjustment set S.
        S = {(v,gamma)} where a pair (v,gamma) represents the vertex v_{t-gamma}.

    :param sg: networkx directed graph
    :param gamma_max: int
    :param x: node
    :param y: node
    :param gamma_xy: int
    :poram gamma_min_dict: dict (edge, int)
    :return: (float, Set (nodes, int))
    """

    assert x != y
    assert (x in sg.nodes) and (y in sg.nodes)
    assert 0 <= gamma_xy
    assert is_identifiable(sg, x, y, gamma_xy, gamma_max)

    if (not sg.has_edge(x, y)) or (gamma_xy > gamma_max):
        print("x not in Parents(sg,y) or gamma_xy > gamma_max so direct effect is zero")
        return (0, set())

    D = nx.descendants(sg, y).union({y})
    A = sg.nodes - D
    adjustment_set = set()
    for v in D:
        for gamma in range(1, gamma_max + 1):
            adjustment_set.add((v, gamma))
    for v in A:
        for gamma in range(0, gamma_max + 1):
            adjustment_set.add((v, gamma))
    try:
        adjustment_set.remove((x, gamma_xy))
    except KeyError:
        pass

    return (None, adjustment_set)


def smaller_adjustment_set_for_direct_effect_in_scg(sg, x, y, gamma_xy, gamma_max = 1, gamma_min_dict = None):
    # todo use gamma_min_dict
    """
    Gives a single-door set to identify the direct effect from x to y with lag gamma_xy if it is identifiable
    for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
    Otherwise fails.
    This corresponds to Proposition 1 in {identifiability of direct from summary causal graphs}.
    The returned value of this function (x,S) can be interpreted as follows:
    if x != None:
        x is the direct effect.
    else x == None:
        The direct effect is can be estimated using the adjustment set S.
        S = {(v,gamma)} where a pair (v,gamma) represents the vertex v_{t-gamma}.

    :param sg: networkx directed graph
    :param gamma_max: int
    :param x: node
    :param y: node
    :param gamma_xy: int
    :poram gamma_min_dict: dict (edge, int)
    :return: (float, Set (nodes, int))
    """
    assert x != y
    assert (x in sg.nodes) and (y in sg.nodes)
    assert 0 <= gamma_xy
    assert is_identifiable(sg, x, y, gamma_xy, gamma_max)

    if (not sg.has_edge(x, y)) or (gamma_xy > gamma_max):
        print("x not in Parents(sg,y) or gamma_xy > gamma_max so direct effect is zero")
        return (0, set())

    D = (nx.ancestors(sg, y).union({y})).intersection(nx.descendants(sg, y).union({y}))
    A = (nx.ancestors(sg, y).union({y})) - D
    adjustment_set = set()
    for v in D:
        for gamma in range(1, gamma_max + 1):
            adjustment_set.add((v, gamma))
    for v in A:
        for gamma in range(0, gamma_max + 1):
            adjustment_set.add((v, gamma))
    try:
        adjustment_set.remove((x, gamma_xy))
    except KeyError:
        pass

    return (None, adjustment_set)

def adjustment_set_for_direct_effect_in_ascgl_using_ParentsY(sg, x, y, gamma_xy, gamma_max = 1, gamma_min_dict = None):
    """
    Gives a single-door set to identify the direct effect from x to y with lag gamma_xy if sg is a ASCGL
    for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
    Otherwise fails.
    This corresponds to a superset of the set described Definition 12 in {Root Cause Identification for Collective Anomalies in Time Series given an Acyclic Summary Causal Graph with Loops}.
    The returned value of this function (x,S) can be interpreted as follows:
    if x != None:
        x is the direct effect.
    else x == None:
        The direct effect is can be estimated using the adjustment set S.
        S = {(v,gamma)} where a pair (v,gamma) represents the vertex v_{t-gamma}.

    :param sg: networkx directed graph
    :param gamma_max: int
    :param x: node
    :param y: node
    :param gamma_xy: int
    :poram gamma_min_dict: dict (edge, int)
    :return: (float, Set (nodes, int))
    """
    assert x != y
    assert (x in sg.nodes) and (y in sg.nodes)
    assert 0 <= gamma_xy
    assert is_ascgl(sg)

    if (not sg.has_edge(x, y)) or (gamma_xy > gamma_max):
        print("x not in Parents(sg,y) or gamma_xy > gamma_max so direct effect is zero")
        return (0, set())
    PY = set(sg.predecessors(y)).difference({x,y})
    adjustment_set = set()
    for v in PY:
        for gamma in range(gamma_min_dict[(v,y)], gamma_max +1):
            adjustment_set.add((v, gamma))
    if ((x,x) in sg.edges) and ((y,y) in sg.edges):
        adjustment_set.update({(y,gamma) for gamma in range(1, gamma_max + 1)})
        adjustment_set.update({(x,gamma) for gamma in range(gamma_min_dict[(x,y)], gamma_xy)})
        adjustment_set.update({(x,gamma) for gamma in range(gamma_xy + 1, gamma_xy + gamma_max + 1)})

    return (None, adjustment_set)

def adjustment_set_for_direct_effect_in_ascgl_using_ParentsXY(sg, x, y, gamma_xy, gamma_max = 1, gamma_min_dict = None):
    """
    Gives a single-door set to identify the direct effect from x to y with lag gamma_xy if sg is a ASCGL
    for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
    Otherwise fails.
    This corresponds to a superset of the set described Definition 12 in {Root Cause Identification for Collective Anomalies in Time Series given an Acyclic Summary Causal Graph with Loops}.
    The returned value of this function (x,S) can be interpreted as follows:
    if x != None:
        x is the direct effect.
    else x == None:
        The direct effect is can be estimated using the adjustment set S.
        S = {(v,gamma)} where a pair (v,gamma) represents the vertex v_{t-gamma}.

    :param sg: networkx directed graph
    :param gamma_max: int
    :param x: node
    :param y: node
    :param gamma_xy: int
    :poram gamma_min_dict: dict (edge, int)
    :return: (float, Set (nodes, int))
    """
    assert x != y
    assert (x in sg.nodes) and (y in sg.nodes)
    assert 0 <= gamma_xy
    assert is_ascgl(sg)

    if (not sg.has_edge(x, y)) or (gamma_xy > gamma_max):
        print("x not in Parents(sg,y) or gamma_xy > gamma_max so direct effect is zero")
        return (0, set())
    PX = set(sg.predecessors(x)).difference({x,y})
    PY = set(sg.predecessors(y)).difference({x,y})
    adjustment_set = set()
    for v in PX:
        for gamma in range(gamma_xy + gamma_min_dict[(v,x)], gamma_xy + gamma_max + 1):
            adjustment_set.add((v, gamma))
    for v in PY:
        for gamma in range(gamma_min_dict[(v,y)], gamma_max +1):
            adjustment_set.add((v, gamma))
    if ((x,x) in sg.edges) and ((y,y) in sg.edges):
        adjustment_set.update({(y,gamma) for gamma in range(1, gamma_xy + 1)})
        adjustment_set.update({(x,gamma) for gamma in range(gamma_min_dict[(x,y)], gamma_xy)})
        adjustment_set.update({(x,gamma) for gamma in range(gamma_xy + 1, gamma_xy + gamma_max + 1)})

    return (None, adjustment_set)
