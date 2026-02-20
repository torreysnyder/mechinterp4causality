import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyro
import random
import matplotlib
from causal_transformers.utils.name_utils import encode_scm_index, get_remapped_node_name


def render_causal_model(model, args, filename):
    print("rendering causal model...")
    pyro.render_model(model, args, render_params=True, filename=filename)
    print("rendering causal model done.")


def custom_layout(G, vertical_spacing=1, horizontal_spacing=1, noise_offset=0.3):
    def add_jitter(pos, amount=0.1):
        return {node: (x + random.uniform(-amount, amount),
                       y + random.uniform(-amount, amount))
                for node, (x, y) in pos.items()}
    non_noise_nodes = [n for n in G.nodes() if not n.startswith('U')]
    noise_nodes = [n for n in G.nodes() if n.startswith('U')]
    H = G.subgraph(non_noise_nodes)
    layers = list(nx.topological_generations(H))
    pos = {}
    max_layer_size = max(len(layer) for layer in layers)
    for i, layer in enumerate(layers):
        layer_size = len(layer)
        for j, node in enumerate(layer):
            x = (j - (layer_size - 1) / 2) * horizontal_spacing
            y = -i * vertical_spacing
            pos[node] = (x, y)

    for noise_node in noise_nodes:
        parent_node = noise_node[1:]
        node_name = "N" + str(parent_node)
        if node_name in pos:
            print(node_name)
            x, y = pos[node_name]
            pos[noise_node] = (x - noise_offset, y + noise_offset)
    return pos


def render_world(cfg,
                 weights,
                 filename=None,
                 figsize=(9, 7),
                 render_weights=False,
                 include_bias=True,
                 render_remapped_names=False,
                 include_noise_terms=False,
                 world_index=None):
    connectivity = (torch.triu(weights, diagonal=1) != 0).numpy()
    G = nx.DiGraph(connectivity)
    node_names = {}
    node_indices = {}
    for node_index in G.nodes():
        node_name = "N" + str(node_index)
        node_names[node_index] = node_name
        node_indices[node_name] = node_index
    G = nx.relabel_nodes(G, node_names)
    if include_noise_terms:
        for node_name in list(G.nodes()):
            node_index = node_indices[node_name]
            noise_node = "U" + str(node_index)
            G.add_node(noise_node)
            G.add_edge(noise_node, node_name)
    pos = custom_layout(G)
    fig, ax = plt.subplots(figsize=figsize)
    if include_noise_terms:
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes() if n.startswith('U')],
                               node_color='#bbbbbb',
                               node_size=4000, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes() if not n.startswith('U')],
                           node_color='black', node_size=4000, ax=ax)
    plt.rcParams['font.family'] = 'helvetica_1'
    if world_index is not None:
        scm_letters = encode_scm_index(cfg, world_index)
        world_name = f"SCM {world_index}"
        plt.text(0.5, 1.1,
                 world_name,
                 horizontalalignment='center',
                 verticalalignment='top',
                 transform=ax.transAxes,
                 fontsize=35,
                 fontweight='bold',
                 zorder=10,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=1.0))
    edge_options = {
        "ax": ax,
        "edge_color": 'black',
        "arrows": True,
        "arrowsize": 30,
        "width": 5.0,
        "connectionstyle": "arc3,rad=0.0",
        "min_source_margin": 20,
        "min_target_margin": 20,
    }
    nx.draw_networkx_edges(G, pos, **edge_options)
    if render_weights:
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
        for source, target in G.edges():
            if source.startswith('U'):
                continue
            weight = weights[node_indices[source], node_indices[target]].item()
            color = "black"
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            w1 = 0.6
            w2 = 1 - w1
            x_mid = x1 * w1 + x2 * w2
            y_mid = y1 * w1 + y2 * w2
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy * 1.1, dx)
            rotation = np.degrees(angle) - 90
            if rotation > 90:
                rotation -= 180
            elif rotation < -90:
                rotation += 180
            ax.text(x_mid, y_mid, f"{weight:.0f}",
                    fontsize=25,
                    rotation=rotation,
                    rotation_mode='anchor',
                    ha='center', va='center',
                    color='white',
                    bbox=dict(facecolor='orange', edgecolor='none', alpha=1.0, pad=5))
        if include_bias:
            for source in G.nodes():
                if source.startswith('U'):
                    continue
                x, y = pos[source]
                weight = weights[node_indices[source], node_indices[source]].item()
                color = "black"
                trans_offset = matplotlib.transforms.ScaledTranslation(0, 0.5, fig.dpi_scale_trans)
                trans = ax.transData + trans_offset
                ax.text(x, y, f"{weight:/0f}",
                        fontsize=25,
                        ha='center', va='center',
                        color="white",
                        transform=trans,
                        bbox=dict(facecolor='orange', edgecolor='none', alpha=1.0, pad=5))
    new_labels = {}
    new_pos = {}
    for node in G.nodes():
        node_int = int(node[1:])
        node_str = str(node_int + 1)
        if node.startswith('U'):
            new_label = r"$\text{U}_{" + node_str + "}$"
        else:
            new_label = r"$\text{V}_{" + node_str + "}$"
            if render_remapped_names:
                assert world_index is not None, "World index must be provided to render remapped names"
                remapped_name = get_remapped_node_name(cfg, world_index, node_int)
                new_label += f"\n({remapped_name})"
        new_labels[node] = new_label
        new_pos[new_label] = pos[node]
    G = nx.relabel_nodes(G, new_labels)
    font_size = 25
    nx.draw_networkx_labels(G, new_pos, ax=ax, font_size=font_size, font_color='white', font_weight="bold", font_family='Helvetica')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_facecolor('#f0f0f0')
    plt.axis('off')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


