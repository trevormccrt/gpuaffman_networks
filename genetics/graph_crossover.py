import copy
import random
from collections import deque
import networkx as nx
import numpy as np


def find_connected_subgraph(graph: nx.MultiDiGraph, max_subgraph_size, starting_nodes):
    # tail, head
    visited = []
    to_visit = deque([(x, None) for x in starting_nodes])
    while len(visited) < max_subgraph_size:
        if not to_visit:
            break
        this_node = to_visit.pop()[0]
        if not this_node in visited:
            visited.append(this_node)
            all_in_edges = graph.in_edges(nbunch=(this_node), keys=True, data=True)
            in_nodes = [x[0] for x in all_in_edges]
            for next_node, edge in zip(in_nodes, all_in_edges):
                if not next_node in visited:
                    to_visit.appendleft((next_node, edge))
            all_out_edges = graph.out_edges(nbunch=(this_node), keys=True, data=True)
            out_nodes = [x[1] for x in all_out_edges]
            for next_node, edge in zip(out_nodes, all_out_edges):
                if not next_node in visited:
                    to_visit.appendleft((next_node, edge))
    cut_edges = []
    for x in to_visit:
        if not ((x[1][0] in visited) and (x[1][1] in visited)):
            cut_edges.append(x[1])
    return visited, cut_edges


def find_subgraphs(graph, subgraph_size, init_starting_nodes):
    starting_nodes = init_starting_nodes
    full_subgraph = []
    full_cut_wires = []
    while True:
        subgraph, cut_wires = find_connected_subgraph(graph, subgraph_size - len(full_subgraph), starting_nodes)
        full_subgraph += subgraph
        full_cut_wires += cut_wires
        if len(full_subgraph) == subgraph_size:
            break
        avail_nodes = []
        for node in graph.nodes:
            if not node in full_subgraph:
                avail_nodes.append(node)
        if not avail_nodes:
            raise RuntimeError("Subgraph size impossible to find")
        starting_nodes = np.random.choice(avail_nodes, (1))

    return full_subgraph, full_cut_wires


def connection_array_to_dict(connections, used_connections, node_labels=None):
    if node_labels is None:
        node_labels = np.arange(start=0, stop=connections.shape[0], step=1)
    in_connections = dict([(i, []) for i in node_labels])
    out_connections = dict([(i, []) for i in node_labels])
    active_ind = np.argwhere(used_connections)
    for idx in active_ind:
        in_connections[node_labels[idx[0]]] = in_connections[node_labels[idx[0]]] + [node_labels[connections[idx[0], idx[1]]]]
        out_connections[node_labels[connections[idx[0], idx[1]]]] = out_connections[node_labels[connections[idx[0], idx[1]]]] + [node_labels[idx[0]]]
    return in_connections, out_connections


def connection_spec_to_graph(connections, used_connections, node_labels):
    g = nx.MultiDiGraph(max_k=connections.shape[1])
    nodes = [(x, {"ordering": y}) for y, x in enumerate(node_labels)]
    g.add_nodes_from(nodes)
    edges = []
    active_connections = np.argwhere(used_connections)
    for connection in active_connections:
        edges.append((node_labels[connections[connection[0], connection[1]]], node_labels[connection[0]] , {"fn_row": connection[1]}))
    g.add_edges_from(edges)
    return g


def graph_to_connection_spec(graph: nx.DiGraph):
    N = graph.number_of_nodes()
    k_max = graph.graph["max_k"]
    position_map = dict([(node[0],node[1]["ordering"]) for node in graph.nodes(data=True)])
    used_connections = np.zeros((N, k_max), dtype=np.bool_)
    connections = np.random.randint(0, N, (N, k_max), dtype=np.uint8)
    for edge in graph.edges(data=True):
        connections[position_map[edge[1]], edge[2]["fn_row"]] = position_map[edge[0]]
        used_connections[position_map[edge[1]], edge[2]["fn_row"]] = True
    return connections, used_connections


def strip_node(graph: nx.MultiDiGraph, nodes_to_strip):
    cut_edges = []
    for edge in graph.edges(keys=True, data=True):
        if (edge[0] in nodes_to_strip) ^ (edge[1] in nodes_to_strip):
            cut_edges.append(edge)
    stripped_graph = copy.deepcopy(graph)
    stripped_graph.remove_nodes_from(nodes_to_strip)
    return stripped_graph, cut_edges


def split_parents(graph_1, graph_2, size_first, size_second, special_nodes):
    first_subgraph, first_cut_wires = find_subgraphs(graph_1, size_first, np.random.choice(list(graph_1.nodes), (1)))
    special_first = []
    seeds_second = []
    for snode in special_nodes:
        if not snode in first_subgraph:
            seeds_second.append(snode)
        else:
            special_first.append(snode)
    pruned_second, removed_special = strip_node(graph_2, special_first)
    second_subgraph, second_cut_wires = find_subgraphs(pruned_second, size_second, seeds_second)
    for item in removed_special:
        if item[0] in second_subgraph or item[1] in second_subgraph:
            second_cut_wires.append(item)
    return first_subgraph, first_cut_wires, second_subgraph, second_cut_wires


def sort_cut_edges(nodes, edges):
    edges_in = []
    edges_out = []
    for edge in edges:
        if edge[0] in nodes:
            edges_out.append(edge)
        else:
            edges_in.append(edge)
    return edges_in, edges_out


def rewire_inputs_random(subgraph_from, edges_in_to, edges_out_from):
    in_nodes_to_replace = list(set([x[0] for x in edges_in_to]))
    out_avail_nodes = [x[0] for x in edges_out_from]
    node_map = {}
    for node_in in in_nodes_to_replace:
        if node_in in subgraph_from:
            node_map[node_in] = node_in
        else:
            node_map[node_in] = np.random.choice(out_avail_nodes)
    return [(node_map[x[0]], *x[1:]) for x in edges_in_to]


def mend_edges_random(subgraph_1, cut_edges_1, subgraph_2, cut_edges_2):
    edges_in_1, edges_out_1 = sort_cut_edges(subgraph_1, cut_edges_1)
    edges_in_2, edges_out_2 = sort_cut_edges(subgraph_2, cut_edges_2)
    new_edges_1 = rewire_inputs_random(subgraph_2, edges_in_1, edges_out_2)
    new_edges_2 = rewire_inputs_random(subgraph_1, edges_in_2, edges_out_1)
    return new_edges_1, new_edges_2


def graph_crossover_random(graph_1, graph_2, special_nodes, size_cut_1):
    N = len(graph_1.nodes)
    size_cut_2 = N - size_cut_1
    first_subgraph, first_cut_wires, second_subgraph, second_cut_wires = split_parents(
        graph_1, graph_2, size_cut_1, size_cut_2, special_nodes)
    new_edges_1, new_edges_2 = mend_edges_random(first_subgraph, first_cut_wires, second_subgraph, second_cut_wires)
    return first_subgraph, new_edges_1, second_subgraph, new_edges_2

