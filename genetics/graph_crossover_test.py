import numpy as np

from genetics import graph_crossover


def test_connection_spec_to_dict():
    for j in range(10):
        N = np.random.randint(0, 30)
        k_max = np.random.randint(0, 8)
        connections = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
        used_connections = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
        in_connections, out_connections = graph_crossover.connection_array_to_dict(connections, used_connections)
        for i in range(np.shape(connections)[0]):
            this_in = []
            this_out = []
            for j in range(np.shape(connections)[1]):
                if used_connections[i, j]:
                    this_in.append(connections[i, j])
            for m in range(np.shape(connections)[0]):
                for n in range(np.shape(connections)[1]):
                    if used_connections[m, n] and connections[m, n] == i:
                        this_out.append(m)
            assert sorted(this_out) == sorted(out_connections[i])
            assert sorted(this_in) == sorted(in_connections[i])


def test_labeled_connection_spec_to_dict():
    for j in range(10):
        N = np.random.randint(0, 30)
        k_max = np.random.randint(0, 8)
        connections = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
        used_connections = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
        shift = np.random.randint(0, 100)
        node_labels = np.arange(start=shift, stop=shift+N, step=1)
        in_connections, out_connections = graph_crossover.connection_array_to_dict(connections, used_connections,
                                                                                   node_labels=node_labels)
        for i in range(np.shape(connections)[0]):
            this_in = []
            this_out = []
            for j in range(np.shape(connections)[1]):
                if used_connections[i, j]:
                    this_in.append(node_labels[connections[i, j]])
            for m in range(np.shape(connections)[0]):
                for n in range(np.shape(connections)[1]):
                    if used_connections[m, n] and connections[m, n] == i:
                        this_out.append(node_labels[m])
            assert sorted(this_out) == sorted(out_connections[node_labels[i]])
            assert sorted(this_in) == sorted(in_connections[node_labels[i]])


def test_find_connected_subgraph_full():
    connections = np.array([[1, 2], [2, 0], [1, 0]])
    used_connnections = np.array([[True, True], [False, False], [False, False]])
    in_cons, out_cons = graph_crossover.connection_array_to_dict(connections, used_connnections)
    subgraph, cut_wires_in, cut_wires_out = graph_crossover.find_connected_subgraph(in_cons, out_cons, 3, [0])
    assert sorted(subgraph) == [0, 1, 2]
    assert not cut_wires_in
    assert not cut_wires_out
    subgraph_l, cut_wires_in_l, cut_wires_out_l = graph_crossover.find_connected_subgraph(in_cons, out_cons, 5, [0])
    assert subgraph_l == subgraph
    assert cut_wires_in_l == cut_wires_in
    assert cut_wires_out_l == cut_wires_out


def test_find_connected_subgraph():
    # tail, head
    conn_dict_in = dict([(0, [1]), (1, [2]), (2, [1]), (3, [])])
    con_dict_out = dict([(0, [2]), (1, [3]), (2, []), (3, [])])
    subgraph, cut_wires_in, cut_wires_out = graph_crossover.find_connected_subgraph(conn_dict_in, con_dict_out, 3, [0])
    assert subgraph == [0, 1, 2]
    assert not cut_wires_in
    assert cut_wires_out == [(1, 3)]

    subgraph_l, cut_wires_in_l, cut_wires_out_l = graph_crossover.find_connected_subgraph(conn_dict_in, con_dict_out, 5, [0])
    assert sorted(subgraph_l) == sorted([0, 1, 2, 3])
    assert not cut_wires_in_l
    assert not cut_wires_out_l


def test_find_big_connected_subgraph():
    conn_dict_in = dict([(0, [1, 2, 3]), (1, [2]), (2, [1]), (3, []), (4, [1, 2, 5, 8]), (5, (1, 5, 4)), (6, [8, 4, 1]), (7, [1, 3, 2]), (8, [0])])
    con_dict_out = dict([(0, [4]), (1, [3]), (2, []), (3, []), (4, [6, 7]), (5, (0, 2, 7)), (6, [0, 1]), (7, [6, 4, 2]), (8, [0, 1])])
    subgraph, cut_wires_in, cut_wires_out = graph_crossover.find_connected_subgraph(conn_dict_in, con_dict_out, 5, [0])
    assert sorted(subgraph) == [0, 1, 2, 3, 4]
    assert set(cut_wires_in) == set([(8, 4), (5, 4)])
    assert set(cut_wires_out) == set([(4, 6), (4, 7)])


def test_find_full_subgraph_connected():
    conn_dict_in = dict([(0, [1]), (1, [2]), (2, [1]), (3, [])])
    con_dict_out = dict([(0, [2]), (1, [3]), (2, []), (3, [])])
    subgraph, cut_wires_in, cut_wires_out = graph_crossover.find_subgraphs(
        conn_dict_in, con_dict_out, 3, [0], np.arange(start=0, stop=4, step=1))
    assert subgraph == [0, 1, 2]
    assert not cut_wires_in
    assert cut_wires_out == [(1, 3)]


def test_find_full_subgraph_islands():
    conn_dict_in = dict([([0, [1]]), (1, [2]), (2, [0]), (3, [4]), (4, [5]), (5, [3])])
    conn_dict_out = dict([(0, [2]), (1, [0]), (2, [1]), (3, [5]), (4, [3]), (5, [4])])

    subgraph, cut_wires_in, cut_wires_out = graph_crossover.find_subgraphs(
        conn_dict_in, conn_dict_out, 6, [0], np.arange(start=0, stop=4, step=1))
    assert len(subgraph) == 6
    assert not cut_wires_in
    assert not cut_wires_out

    subgraph, cut_wires_in, cut_wires_out = graph_crossover.find_connected_subgraph(
        conn_dict_in, conn_dict_out, 6, [0])
    assert sorted(subgraph) == [0, 1, 2]


def test_strip_node():
    N = 20
    k_max = 8
    connections = np.random.randint(0, N, (N, k_max)).astype(np.uint8)
    used_connections = np.random.binomial(1, 0.5, (N, k_max)).astype(np.bool_)
    in_connections, _ = graph_crossover.connection_array_to_dict(connections, used_connections)
    to_strip = [7]
    removed_in_connections, stripped_in_connections = graph_crossover.strip_node(in_connections, to_strip)
    assert not np.any([x in stripped_in_connections.keys() for x in to_strip])
    for x in to_strip:
        assert not np.any([x in v for _, v in stripped_in_connections.items()])


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

        first_parent_in, first_parent_out = graph_crossover.connection_array_to_dict(
            np.random.randint(0, N, (N, k_max)), np.ones((N, k_max)).astype(np.bool_), node_labels=nodes_first)

        second_parent_in, second_parent_out = graph_crossover.connection_array_to_dict(
            np.random.randint(0, N, (N, k_max)), np.ones((N, k_max)).astype(np.bool_), node_labels=nodes_second)

        first_subgraph, first_cut_wires_in, first_cut_wires_out, \
            second_subgraph, second_cut_wires_in, second_cut_wires_out = graph_crossover.split_parents(
            nodes_first, first_parent_in, first_parent_out, nodes_second, second_parent_in,
            second_parent_out, first_parent_size, second_parent_size, special_nodes)
        assert len(first_subgraph + second_subgraph) == N
        assert not np.any([x in second_subgraph for x in first_subgraph])
        for x in special_nodes:
            assert (x in first_subgraph or x in second_subgraph) and not (x in first_subgraph and x in second_subgraph)
        assert [x[1] in first_subgraph for x in first_cut_wires_in]
        assert [x[0] in first_subgraph for x in first_cut_wires_out]
        assert [x[1] in second_subgraph for x in second_cut_wires_in]
        assert [x[0] in second_subgraph for x in second_cut_wires_out]


def test_simple_wiring():
    graph_con_1 = np.array([[1, 2], [0, 3], [4, 0], [1, 2], [5, 3], [4, 6], [4, 5]])
    graph_used_con_1 = np.array(
        [[True, True], [True, True], [True, True], [True, True], [True, True], [True, True], [True, True]])
    graph_con_2 = np.array([[1, 5], [4, 3], [4, 1], [4, 2], [5, 6], [1, 6], [4, 2]])
    graph_used_con_2 = np.array(
        [[True, True], [True, True], [True, True], [True, True], [True, True], [True, True], [True, True]])
    N_breed_test = graph_con_1.shape[0]
    # %%
    special_nodes = [0, 1, 2]
    first_node_labels = np.arange(start=0, stop=N_breed_test, step=1)
    second_node_labels = np.arange(start=N_breed_test, stop=2 * N_breed_test, step=1)
    second_node_labels[:len(special_nodes)] = special_nodes
    # %%
    first_in, first_out = graph_crossover.connection_array_to_dict(graph_con_1, graph_used_con_1, first_node_labels)
    second_in, second_out = graph_crossover.connection_array_to_dict(graph_con_2, graph_used_con_2, second_node_labels)
    size_first = 4
    size_second = N_breed_test - size_first
    first_subgraph, first_cut_wires_in, first_cut_wires_out, \
        second_subgraph, second_cut_wires_in, second_cut_wires_out = graph_crossover.split_parents(first_node_labels,
                                                                                                   first_in, first_out,
                                                                                                   second_node_labels,
                                                                                                   second_in,
                                                                                                   second_out,
                                                                                                   size_first,
                                                                                                   size_second,
                                                                                                   special_nodes)
    graph_crossover.mend_cut_wires(first_subgraph, first_cut_wires_in, first_cut_wires_out,
                                  second_subgraph, second_cut_wires_in, second_cut_wires_out)
