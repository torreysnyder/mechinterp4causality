from .dataset_utils import get_d_vocab, get_n_ctx
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def get_hooked_transformer_model(cfg, device):
    d_vocab = get_d_vocab(cfg)
    n_ctx = get_n_ctx(cfg)
    ht_cfg = HookedTransformerConfig(
        **cfg.model.hooked_transformer,
        d_vocab=d_vocab,
        n_ctx=n_ctx,
        init_weights=True,
        device=device,
        seed=None
    )
    return HookedTransformer(ht_cfg)
