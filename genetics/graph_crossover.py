import copy
import random
from collections import deque
import networkx as nx
import numpy as np


def find_connected_subgraph(in_connections, out_connections, max_subgraph_size, starting_nodes):
    # tail, head
    visited = []
    to_visit = deque([(x, -1, -1) for x in starting_nodes])
    while len(visited) < max_subgraph_size:
        if not to_visit:
            break
        this_node = to_visit.pop()[0]
        if not this_node in visited:
            visited.append(this_node)
            for next_node in in_connections[this_node]:
                if not next_node in visited:
                    to_visit.appendleft((next_node, this_node, 0))
            for next_node in out_connections[this_node]:
                if not next_node in visited:
                    to_visit.appendleft((next_node, this_node, 1))
    cut_wires_out = []
    cut_wires_in = []
    for x in to_visit:
        if not x[0] in visited and x[1] in visited:
            if x[-1]:
                cut_wires_out.append(x[:-1][::-1])
            else:
                cut_wires_in.append(x[:-1])
    return visited, cut_wires_in, cut_wires_out


def find_subgraphs(in_connections, out_connections, subraph_size, init_starting_nodes, all_nodes):
    starting_nodes = init_starting_nodes
    full_subgraph = []
    full_cut_wires_in = []
    full_cut_wires_out = []
    while True:
        subgraph, cut_wires_in, cut_wires_out = find_connected_subgraph(in_connections, out_connections, subraph_size-len(full_subgraph), starting_nodes)
        full_subgraph += subgraph
        full_cut_wires_in += cut_wires_in
        full_cut_wires_out += cut_wires_out
        if len(full_subgraph) == subraph_size:
            break
        avail_nodes = []
        for node in all_nodes:
            if not node in full_subgraph:
                avail_nodes.append(node)
        if not avail_nodes:
            raise RuntimeError("Subgraph size impossible to find")
        starting_nodes = np.random.choice(avail_nodes, (1))

    return full_subgraph, full_cut_wires_in, full_cut_wires_out


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


def strip_node(connections_dict, to_strip):
    removed_connections = []
    for x in to_strip:
        removed_connections += [(x, j) for j in connections_dict[x]]
    stripped_connections_dict = copy.deepcopy(connections_dict)
    [stripped_connections_dict.pop(x) for x in to_strip]
    for key, value in stripped_connections_dict.items():
        keep_values = []
        for x in value:
            if not x in to_strip:
                keep_values.append(x)
        stripped_connections_dict[key] = keep_values
    return removed_connections, stripped_connections_dict


def split_parents(nodes_1, in_connections_1, out_connections_1, nodes_2, in_connections_2, out_connections_2, size_first, size_second, special_nodes):
    first_subgraph, first_cut_wires_in, first_cut_wires_out = find_subgraphs(
        in_connections_1, out_connections_1, size_first, np.random.choice(nodes_1, (1)), nodes_1)
    special_first = []
    seeds_second = []
    for snode in special_nodes:
        if not snode in first_subgraph:
            seeds_second.append(snode)
        else:
            special_first.append(snode)
    deleted_in, stripped_second_in = strip_node(in_connections_2, special_first)
    deleted_out, stripped_second_out = strip_node(out_connections_2, special_first)
    avail_nodes_2 = []
    for node in nodes_2:
        if not node in special_first:
            avail_nodes_2.append(node)
    second_cut_wires_in = []
    second_cut_wires_out = []
    second_cut_wires_out += [x[::-1] for x in deleted_in]
    second_cut_wires_in += [x for x in deleted_out]
    second_subgraph, second_cut_wires_in, second_cut_wires_out = find_subgraphs(
        stripped_second_in, stripped_second_out, size_second, seeds_second, avail_nodes_2)
    for item in deleted_in:
        if item[1] in second_subgraph:
            second_cut_wires_out += [item[::-1]]
    for item in deleted_out:
        if item[1] in second_subgraph:
            second_cut_wires_in += [item]
    return first_subgraph, first_cut_wires_in, first_cut_wires_out, \
        second_subgraph, second_cut_wires_in, second_cut_wires_out


def match_wiring(subgraph_from, subgraph_to, to_in, from_out):
    num_ports_in = {}
    for item in subgraph_to:
        num_ports_in[item] = np.sum([x[1] == item for x in to_in])
    new_edges = []
    for item in to_in:
        from_node = item[0]
        to_node = item[1]
        if from_node in subgraph_from and num_ports_in[to_node]>0:
            num_ports_in[to_node] -= 1
            new_edges.append(item)
    print("")
    return new_edges




def mend_cut_wires(first_subgraph, first_cut_wires_in, first_cut_wires_out,
                   second_subgraph, second_cut_wires_in, second_cut_wires_out):
    edges_in_first = match_wiring(second_subgraph, first_subgraph, first_cut_wires_in, second_cut_wires_out)
    edges_in_second = match_wiring(first_subgraph, second_subgraph, second_cut_wires_in, first_cut_wires_out)
    return edges_in_first + edges_in_second
