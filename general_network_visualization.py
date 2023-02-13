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
