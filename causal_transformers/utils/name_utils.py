import numpy as np
import string
import torch


def encode_scm_index(cfg, scm_index, alphabet=string.ascii_uppercase):
    width = cfg.dataset.names.scm_code_width
    base = len(alphabet)
    if scm_index < 0:
        raise ValueError("Number must be non-negative")
    max_representable = base ** width - 1
    if scm_index > max_representable:
        raise ValueError(f"Number too large to represent in width {width}")
    result = []
    for _ in range(width):
        scm_index, rem = divmod(scm_index, base)
        result.append(alphabet[rem])
    return ''.join(reversed(result))


def decode_scm_index(string, alphabet=string.ascii_uppercase):
    base = len(alphabet)
    num = 0
    for char in string:
        num = num * base + alphabet.index(char)
    return num


def get_node_name_from_index(cfg, node_index):
    return f"V{(node_index+1):0{cfg.dataset.names.node_name_zero_pad}d}"


def get_scm_name_from_index(cfg, scm_index):
    return f"SCM{scm_index:0{cfg.dataset.names.scm_name_zero_pad}d}"


def reproducible_node_index_remap(cfg, world_index, node_index):
    rng = np.random.default_rng(world_index + 1234)
    permutation = rng.permutation(cfg.dataset.world.n_nodes.max)
    return permutation[node_index]


def get_remapped_node_name(cfg, world_index, node_index):
    if cfg.dataset.remap_variable_names:
        node_index_remapped = reproducible_node_index_remap(cfg, world_index, node_index)
    else:
        node_index_remapped = node_index
    node_name = get_node_name_from_index(cfg, node_index_remapped)
    return node_name



