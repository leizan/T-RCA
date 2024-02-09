"""
Coded by Simon Ferreira and Charles Assaad
"""

import networkx as nx
import matplotlib.pyplot as plt
# -----------------tools------------------------


def is_active(sg, path, adjustment_set = set()):
    """
    Determines whether a path is active in a (cyclic) summary causal graph sg when adjusting on adjustment_set.
    :param sg: networkx directed graph
    :param path: List nodes
    :param adjustment_set: Set nodes
    :return: bool
    """
    assert is_subset(path, sg.nodes)
    assert is_subset(adjustment_set, sg.nodes)

    colliders = set()
    has_seen_right_arrow = False
    for i in range(len(path) - 1):
        if (sg.has_edge(path[i], path[i + 1])) and (sg.has_edge(path[i + 1], path[i])):         #V^i <-> V^{i+1}
            pass
        elif (sg.has_edge(path[i], path[i + 1])) and (not sg.has_edge(path[i + 1], path[i])):   #V^i  -> V^{i+1}
            if (i>0) and (path[i] in adjustment_set):
                return False
            has_seen_right_arrow = True
        elif (not sg.has_edge(path[i], path[i + 1])) and (sg.has_edge(path[i + 1], path[i])):   #V^i <-  V^{i+1}
            if has_seen_right_arrow:
                colliders.add(path[i])
                has_seen_right_arrow = False
        else:                                                                                   #V^i     V^{i+1}
            raise ValueError("Path is not a path in sg: " + str(path[i]) + " and " + str(path[i + 1]) + " are not connected.")
    for c in colliders:
        for d in nx.descendants(sg, c).union({c}):
            if d in adjustment_set:
                break
        else:
            return False
    return True


def is_subset(small_iterable, big_iterable):
    """
    Determines whether small_iterable is a subset of big_iterable.
    :param small_iterable: iterable nodes
    :param big_iterable: iterable nodes
    :return: bool
    """

    for v in small_iterable:
        if v not in big_iterable:
            return False
    return True


def is_non_direct(sg, path):
    """
    Determines whether path is non-direct. (path != X->Y ?)
    :param sg: networkx directed graph
    :param path: List nodes
    :return: bool
    """
    assert is_subset(path, sg.nodes)
    assert len(path) >= 2
    assert path[0] != path[-1]

    if len(path) > 2:
        return True

    x = path[0]
    y = path[-1]
    if sg.has_edge(y, x):
        return True
    if not sg.has_edge(x, y):
        return False

    return True


def exists_cycle(sg, source, forbidden_vertices = []):
    """
    Determines whether there exists a directed cycle from source to source in sg which avoids forbidden_vertices.
    :param sg: networkx directed graph
    :param source: node
    :param forbidden_vertices: List nodes
    :return: bool
    """
    visited = set(forbidden_vertices)
    end_points = set(sg.successors(source)).difference(visited)
    while end_points:
        v = end_points.pop()
        if v == source:
            return True
        visited.add(v)
        end_points.update(set(sg.successors(v)).difference(visited))
    return False


def remove_self_loops(sg):
    """
    Returns a copy of the graph without any self-loops.
    :param sg: networkx directed graph
    :return: networkx directed graph
    """
    sg2 = sg.copy()
    sg2.remove_edges_from(nx.selfloop_edges(sg2))
    return sg2


def is_ascgl(sg):
    """
    Determines whether sg is an ASCGL.
    :param sg: networkx directed graph
    :return: bool
    """
    return nx.is_directed_acyclic_graph(remove_self_loops(sg))


def draw_graph(g, node_size=300):
    pos = nx.spring_layout(g, k=0.25, iterations=25)
    nx.draw(g, pos, with_labels=True, font_weight='bold', node_size=node_size)
    plt.show()
