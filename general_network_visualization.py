import celluloid
import networkx as nx
import numpy as np


def graph_from_spec(connections):
    edges = []
    for node in range(np.shape(connections)[0]):
        for connection in connections[node, :]:
            edges.append((node, connection))
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def graph_from_ragged_spec(connections, used_connections):
    g = nx.DiGraph()
    g.add_nodes_from(np.arange(start=0, stop=np.shape(connections)[0], step=1))
    for i, (c, u) in enumerate(zip(connections, used_connections)):
        this_cons = np.squeeze(c[np.argwhere(u == 1)], -1)
        g.add_edges_from([(i, x) for x in this_cons])
    return g


def plot_graph_with_state(g, layout, state, ax, **kwargs):
    color_list = ["C0" if x else "C1" for x in state]
    nx.draw(g, ax=ax, pos=layout, node_color=color_list, **kwargs)


def graph_animation(g, layout, trajectory, fig, ax, **kwargs):
    camera = celluloid.Camera(fig)
    for state in trajectory:
        plot_graph_with_state(g, layout, state, ax, **kwargs)
        camera.snap()
    animation = camera.animate()
    return animation
