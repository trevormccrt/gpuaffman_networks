import copy

import networkx as nx
import numpy as np

from genetics import graph_crossover


def test_connection_spec_to_graph():
    for j in range(10):
        N = np.random.randint(10, 30)
        k_max = np.random.randint(5, 8)
        connections = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
        used_connections = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
        node_labels = np.arange(start=0, stop=N, step=1)
        np.random.shuffle(node_labels)
        g = graph_crossover.connection_spec_to_graph(connections, used_connections, node_labels)
        assert g.graph["max_k"] == k_max
        n_active_conections = np.sum(used_connections)
        edges = list(g.edges(data=True))
        assert n_active_conections == len(edges)
        for edge in edges:
            index_from = g.nodes[edge[0]]["ordering"]
            index_to = g.nodes[edge[1]]["ordering"]
            assert connections[index_to, edge[2]["fn_row"]] == index_from


def test_graph_to_connection_spec():
    for j in range(10):
        N = np.random.randint(10, 30)
        k_max = np.random.randint(4, 8)
        connections = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
        used_connections = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
        node_labels = np.arange(start=0, stop=N, step=1)
        np.random.shuffle(node_labels)
        g = graph_crossover.connection_spec_to_graph(connections, used_connections, node_labels)
        found_connections, found_used_connections = graph_crossover.graph_to_connection_spec(g)
        np.testing.assert_equal(used_connections, found_used_connections)
        np.testing.assert_equal(connections * used_connections, found_connections * found_used_connections)


def test_find_connected_subgraph_full():
    g = nx.MultiDiGraph()
    g.add_nodes_from([0, 1, 2])
    g.add_edges_from([(0, 1), (0, 2), (1, 2)])
    subgraph, cut_edges = graph_crossover.find_connected_subgraph(g, 3, [0])
    assert sorted(subgraph) == [0, 1, 2]
    assert not cut_edges
    subgraph_l, cut_edges_l = graph_crossover.find_connected_subgraph(g, 5, [0])
    assert subgraph_l == subgraph
    assert cut_edges_l == cut_edges


def test_multiedge_subgraph():
    g = nx.MultiDiGraph()
    g.add_nodes_from([0, 1, 2])
    edges = [(0, 1, 0, {"label": 0}), (0, 1, 1, {"label": 1}),
                      (1, 0, 0, {"label": 0}), (0, 2, 0, {"label": 0}),
                      (1, 2, 0, {"label": 0})]
    g.add_edges_from(edges)
    subgraph, cut_edges = graph_crossover.find_connected_subgraph(g, 1, [0])
    assert sorted(subgraph) == [0]
    assert np.all([x in cut_edges for x in edges[:-1]])


def test_find_connected_subgraph():
    # tail, head
    nodes = [0, 1, 2, 3, 4]
    g = nx.MultiDiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 2), (2, 4)])

    subgraph, cut_edges = graph_crossover.find_connected_subgraph(g, 3, [0])
    assert sorted(subgraph) == [0, 1, 2]
    assert set([x[:2] for x in cut_edges]) == {(2, 4), (3, 2)}

    subgraph_l, cut_edges = graph_crossover.find_connected_subgraph(g, 5, [0])
    assert sorted(subgraph_l) == nodes
    assert not cut_edges


def test_find_full_subgraph_connected():
    g = nx.MultiDiGraph()
    g.add_nodes_from([0, 1, 2])
    g.add_edges_from([(0, 1), (0, 2), (1, 2)])
    subgraph, cut_edges = graph_crossover.find_subgraphs(g, 3, [0])
    assert sorted(subgraph) == [0, 1, 2]
    assert not cut_edges


def test_find_full_subgraph_islands():
    g = nx.MultiDiGraph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5])
    g.add_edges_from([(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (5, 3)])
    subgraph, cut_edges = graph_crossover.find_subgraphs(g, 6, [0])
    assert sorted(subgraph) == [0, 1, 2, 3, 4, 5]
    assert not cut_edges


def test_strip_node():
    g = nx.MultiDiGraph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 0), (3, 0)])
    stripped_graph, cut_edges = graph_crossover.strip_node(g, [1, 3])
    assert set(cut_edges) == {(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)}
    assert sorted(stripped_graph.nodes) == [0, 2]


def _compare_graphs(g1, g2):
    assert sorted(g1.nodes) == sorted(g2.nodes)
    edges_new = sorted(list(g1.edges(keys=True, data=True)), key=lambda x: x[0])
    edges_orig = sorted(list(g2.edges(keys=True, data=True)), key=lambda x: x[0])
    assert len(edges_new) == len(edges_orig)
    for edge in edges_new:
        assert edge in edges_orig


def test_split_parents():
    for i in range(50):
        N = np.random.randint(10, 40)
        k_max = np.random.randint(4, 8)
        special_nodes = [0, 1, 2]
        first_parent_size = np.random.randint(int(N/4), int(N/2))
        second_parent_size = N - first_parent_size

        nodes_first = np.arange(start=0, stop=N, step=1)
        nodes_second = np.arange(start=N, stop=2 * N, step=1)
        nodes_second[:len(special_nodes)] = special_nodes

        connections_1 = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
        used_connections_1 = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
        graph_1 = graph_crossover.connection_spec_to_graph(connections_1, used_connections_1, nodes_first)

        connections_2 = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
        used_connections_2 = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
        graph_2 = graph_crossover.connection_spec_to_graph(connections_2, used_connections_2, nodes_second)

        first_subgraph, first_cut_wires, second_subgraph, second_cut_wires =graph_crossover.split_parents(
            graph_1, graph_2, first_parent_size, second_parent_size, special_nodes)

        assert len(first_subgraph + second_subgraph) == N
        assert not np.any([x in second_subgraph for x in first_subgraph])
        for x in special_nodes:
            assert (x in first_subgraph or x in second_subgraph) and not (x in first_subgraph and x in second_subgraph)
        assert np.all([((x[0] in first_subgraph) ^ (x[1] in first_subgraph)) for x in first_cut_wires])
        assert np.all([((x[0] in second_subgraph) ^ (x[1] in second_subgraph)) for x in second_cut_wires])

        nx_first_subgraph = graph_1.subgraph(first_subgraph)
        nx_first_else = copy.deepcopy(graph_1)
        nx_first_else.remove_nodes_from(first_subgraph)
        equiv_graph_1 = nx.MultiDiGraph()
        equiv_graph_1.add_nodes_from(list(nx_first_subgraph.nodes(data=True)) + list(nx_first_else.nodes(data=True)))
        equiv_graph_1.add_edges_from(list(nx_first_subgraph.edges(data=True, keys=True)) + list(
            nx_first_else.edges(data=True, keys=True)) + first_cut_wires)
        _compare_graphs(graph_1, equiv_graph_1)


        nx_second_subgraph = graph_2.subgraph(second_subgraph)
        nx_second_else = copy.deepcopy(graph_2)
        nx_second_else.remove_nodes_from(second_subgraph)
        equiv_graph_2 = nx.MultiDiGraph()
        equiv_graph_2.add_nodes_from(list(nx_second_subgraph.nodes(data=True)) + list(nx_second_else.nodes(data=True)))
        equiv_graph_2.add_edges_from(list(nx_second_subgraph.edges(data=True, keys=True)) + list(
            nx_second_else.edges(data=True, keys=True)) + second_cut_wires)
        _compare_graphs(graph_2, equiv_graph_2)

#
#
# def test_simple_wiring():
#     graph_con_1 = np.array([[1, 2], [0, 3], [4, 0], [1, 2], [5, 3], [4, 6], [4, 5]])
#     graph_used_con_1 = np.array(
#         [[True, True], [True, True], [True, True], [True, True], [True, True], [True, True], [True, True]])
#     graph_con_2 = np.array([[1, 5], [4, 3], [4, 1], [4, 2], [5, 6], [1, 6], [4, 2]])
#     graph_used_con_2 = np.array(
#         [[True, True], [True, True], [True, True], [True, True], [True, True], [True, True], [True, True]])
#     N_breed_test = graph_con_1.shape[0]
#     # %%
#     special_nodes = [0, 1, 2]
#     first_node_labels = np.arange(start=0, stop=N_breed_test, step=1)
#     second_node_labels = np.arange(start=N_breed_test, stop=2 * N_breed_test, step=1)
#     second_node_labels[:len(special_nodes)] = special_nodes
#     # %%
#     first_in, first_out = graph_crossover.connection_array_to_dict(graph_con_1, graph_used_con_1, first_node_labels)
#     second_in, second_out = graph_crossover.connection_array_to_dict(graph_con_2, graph_used_con_2, second_node_labels)
#     size_first = 4
#     size_second = N_breed_test - size_first
#     first_subgraph, first_cut_wires_in, first_cut_wires_out, \
#         second_subgraph, second_cut_wires_in, second_cut_wires_out = graph_crossover.split_parents(first_node_labels,
#                                                                                                    first_in, first_out,
#                                                                                                    second_node_labels,
#                                                                                                    second_in,
#                                                                                                    second_out,
#                                                                                                    size_first,
#                                                                                                    size_second,
#                                                                                                    special_nodes)
#     graph_crossover.mend_cut_wires(first_subgraph, first_cut_wires_in, first_cut_wires_out,
#                                   second_subgraph, second_cut_wires_in, second_cut_wires_out)
#
#
test_split_parents()