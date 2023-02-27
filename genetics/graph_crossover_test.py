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
    assert set([x[:3] for x in cut_edges]) == {(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)}
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


def test_sort_edges():
    nodes = [0, 1, 2, 3]
    all_edges = [(0, 5), (4, 1), (3, 7), (8, 2)]
    edges_in, edges_out = graph_crossover.sort_cut_edges(nodes, all_edges)
    assert set(edges_in) == {(4, 1), (8, 2)}
    assert set(edges_out) == {(0, 5), (3, 7)}


def test_rewire_inputs_random():
    subgraph_from = [6, 7, 8, 9]
    rewire_edges_in = [(6, 0, 0, {"data": 0}), (7, 1, 0, {"data": 1}), (11, 3, 0, {"data": 2}), (12, 5, 0, {"data": 3}),
                       (12, 3, 0, {"data": 4}), (11, 2, 0, {"data": 5})]
    rewire_edges_out = [(9, 3, 0, {"data": 0}), (8, 2, 0, {"data": 1}), (7, 1, 0, {"data": 2}), (6, 3, 0, {"data": 3}),
                        (6, 4, 0, {"data": 4}), (6, 3, 0, {"data": 5})]
    new_edges = graph_crossover.rewire_inputs_random(subgraph_from, rewire_edges_in, rewire_edges_out)
    possible_new_node_choices = [x[0] for x in rewire_edges_out]
    detected_remap_dict = {}
    for old_edge, new_edge in zip(rewire_edges_in, new_edges):
        if old_edge[0] in subgraph_from:
            assert new_edge[0] == old_edge[0]
        else:
            assert new_edge[0] in possible_new_node_choices
            if old_edge[0] in detected_remap_dict:
                assert new_edge[0] == detected_remap_dict[old_edge[0]]
            else:
                detected_remap_dict[old_edge[0]] = new_edge[0]
        assert old_edge[1:] == new_edge[1:]


def test_mend_edges_random():
    subgraph_1 = [0, 1, 2, 3, 4]
    subgraph_2 = [10, 11, 12, 13, 14]
    cut_edges_1 = ((5, 0), (0, 7), (10, 2), (4, 9))
    cut_edges_2 = [(5, 10), (5, 11), (3, 12), (12, 0), (13, 7), (14, 5)]
    new_edges_1, new_edges_2 = graph_crossover.mend_edges_random(subgraph_1, cut_edges_1, subgraph_2, cut_edges_2)
    assert len(new_edges_1) == 2
    assert len(new_edges_2) == 3
    assert np.all([x[0] in subgraph_2 for x in new_edges_1])
    assert np.all([x[1] in subgraph_1 for x in new_edges_1])
    assert np.all([x[0] in subgraph_1 for x in new_edges_2])
    assert np.all([x[1] in subgraph_2 for x in new_edges_2])
    for edge in cut_edges_1:
        if edge[1] in subgraph_1 and edge[0] in subgraph_2:
            assert edge in new_edges_1
    for edge in cut_edges_2:
        if edge[1] in subgraph_2 and edge[0] in subgraph_1:
            assert edge in new_edges_2


def test_merge_indices():
    N = 20
    special_nodes = [0, 1, 2]
    org_1_indices = np.arange(start=0, stop=N, step=1)
    org_2_indices = np.arange(start=N, stop=2*N, step=1)
    fake_nodes_1 = [(x, {"ordering": np.argwhere(org_1_indices == x)[0][0], "org_num":0}) for x in org_1_indices]
    fake_nodes_2 = [(x, {"ordering": np.argwhere(org_2_indices == x)[0][0], "org_num": 1}) for x in org_2_indices]
    update_dicts, org_0_map, org_1_map = graph_crossover.merge_pair_ordering(fake_nodes_1 + fake_nodes_2, special_nodes)
    used_ind = []
    for key, value in update_dicts.items():
        if key in special_nodes:
            assert value["ordering"] == key
        else:
            assert not value["ordering"] in used_ind
        used_ind.append(value["ordering"])
        if key in org_1_indices:
            assert org_0_map[key] == value["ordering"]
        if key in org_2_indices:
            assert org_1_map[np.argwhere(org_2_indices == key)[0][0]] == value["ordering"]


def test_network_crossover_random():
    for i in range(50):
        N = np.random.randint(10, 30)
        k_max = np.random.randint(5, 8)
        special_nodes = [0, 1, 2]
        connections_1 = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
        used_connections_1 = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
        connections_2 = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
        used_connections_2 = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
        new_connections, new_used_connections, org_0_map, org_1_map = graph_crossover.network_crossover_random(
            connections_1, used_connections_1, connections_2,used_connections_2, special_nodes, int(N/3))
        for key, value in org_0_map.items():
            assert np.all(used_connections_1[key] == new_used_connections[value])
        for key, value in org_1_map.items():
            assert np.all(np.equal(used_connections_2[key], new_used_connections[value]))


def test_merge_functions():
    N = np.random.randint(10, 30)
    k_max = np.random.randint(5, 8)
    functions_1 = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
    functions_2 = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
    org_1_stop = int(N/2)
    org_1_keep = np.arange(start=0, stop=org_1_stop, step=1)
    org_2_keep = np.arange(start=org_1_stop, stop=N, step=1)
    ordering = np.arange(start=0, stop=N, step=1)
    np.random.shuffle(ordering)
    org_1_map = dict(zip(org_1_keep, ordering[:len(org_1_keep)]))
    org_2_map = dict(zip(org_2_keep, ordering[len(org_1_keep):]))
    new_functions = graph_crossover.merge_functions(functions_1, functions_2, org_1_map, org_2_map)
    for key, value in org_1_map.items():
        assert np.all(new_functions[value] == functions_1[key])
    for key, value in org_2_map.items():
        assert np.all(new_functions[value] == functions_2[key])
