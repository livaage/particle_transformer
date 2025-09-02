# particle_transformer_mup_local.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

# -------------------------
# MuP building blocks
# -------------------------
class MuPLinear(nn.Module):
    """
    Hidden-weight MuP linear:
      - store weight ~ N(0, 1)
      - forward scales weight by 1/sqrt(in_features)
    Effective weight std = 1/sqrt(in)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._scale = math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)  # unscaled
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # single place of scaling: divide weight by sqrt(in)
        return F.linear(x, self.weight / self._scale, self.bias)


class MuPEmbedding(nn.Module):
    """
    Embedding-like parameters should be initialized with std = 1/sqrt(width)
    and not scaled in forward.
    Use this for token/feature embeddings or initial projection matrices.
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(embedding_dim))

    def forward(self, x):
        # assume nn.Embedding-like usage; x are indices or we use F.linear for continuous inputs
        # For generality, support F.linear if x is float features: x @ weight.T
        if x.dtype in (torch.long, torch.int):
            return F.embedding(x, self.weight)
        else:
            return F.linear(x, self.weight.t())  # if used as a projection


class MuPReadout(nn.Module):
    """
    Readout layer: treated specially for μP. Zero-initialized recommended.
    Acts like a linear layer but zero-init so readout starts close to 0.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class MuPMultiheadAttention(nn.Module):
    """
    Multi-head attention implementing the μP-required attention scaling:
      logits = (q @ k.T) / d_k   (not sqrt(d_k))
    Assumes q,k,v projections are MuPLinear (hidden weights).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout

        # projections as Hidden (MuP) linears
        self.q_proj = MuPLinear(embed_dim, embed_dim, bias=True)
        self.k_proj = MuPLinear(embed_dim, embed_dim, bias=True)
        self.v_proj = MuPLinear(embed_dim, embed_dim, bias=True)
        self.out_proj = MuPLinear(embed_dim, embed_dim, bias=True)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False):
        # query/key/value: (seq, batch, embed) or (batch, seq, embed) if batch_first
        if self.batch_first:
            # transpose to (seq, batch, embed)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        seq_len, batch_size, _ = query.shape

        q = self.q_proj(query)  # (seq, batch, embed)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # reshape for heads -> (heads, batch, seq, head_dim)
        def split_heads(x):
            s, b, e = x.shape
            x = x.view(s, b, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
            return x

        qh = split_heads(q)
        kh = split_heads(k)
        vh = split_heads(v)

        # attention logits scaled by d_k (muP)
        logits = (qh @ kh.transpose(-2, -1)) / float(self.head_dim)  # shape (heads, batch, seq, seq)

        if attn_mask is not None:
            # Handle different attn_mask shapes
            if attn_mask.dim() == 3:
                # attn_mask shape: (batch*heads, seq, seq) or (heads, seq, seq) or (seq, seq)
                if attn_mask.size(0) == batch_size * self.num_heads:
                    # (batch*heads, seq, seq) -> reshape to (heads, batch, seq, seq)
                    attn_mask = attn_mask.view(self.num_heads, batch_size, seq_len, seq_len)
                elif attn_mask.size(0) == self.num_heads:
                    # (heads, seq, seq) -> expand to (heads, batch, seq, seq)
                    attn_mask = attn_mask.unsqueeze(1).expand(-1, batch_size, -1, -1)
                else:
                    # (seq, seq) -> expand to (heads, batch, seq, seq)
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(self.num_heads, batch_size, -1, -1)
            elif attn_mask.dim() == 2:
                # (seq, seq) -> expand to (heads, batch, seq, seq)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(self.num_heads, batch_size, -1, -1)
            
            logits = logits + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq) True for padding positions
            kp = key_padding_mask.unsqueeze(0).unsqueeze(2)  # (1, batch, 1, seq)
            logits = logits.masked_fill(kp, float('-inf'))

        weights = F.softmax(logits, dim=-1)
        if self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout, training=self.training)

        out_h = weights @ vh  # (heads, batch, seq, head_dim)

        # combine heads back to (seq, batch, embed)
        out = out_h.permute(2, 1, 0, 3).contiguous().view(seq_len, batch_size, self.embed_dim)

        if self.batch_first:
            out = out.transpose(0, 1)

        out = self.out_proj(out)
        if need_weights:
            # return average attention weights over heads (batch_first False expected)
            return out, weights.mean(dim=0)
        else:
            return out, None


# -------------------------
# Utilities to convert ParticleTransformer
# -------------------------
def replace_linears_with_mup(module):
    """
    Recursively replace nn.Linear with MuPLinear.
    Keeps other modules intact.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new = MuPLinear(child.in_features, child.out_features, bias=child.bias is not None)
            setattr(module, name, new)
        else:
            replace_linears_with_mup(child)


def replace_attention_modules(module):
    """
    Replace known attention modules with MuPMultiheadAttention.
    Heuristic: if a child module name or classname contains 'attention' or 'attn'
    and it looks like a multihead attention, swap it. Adjust to your repo.
    """
    for name, child in list(module.named_children()):
        clsname = child.__class__.__name__.lower()
        if ('multihead' in clsname) or ('attention' in clsname) or ('attn' in name.lower()):
            # try to detect embed_dim and num_heads
            embed_dim = getattr(child, 'embed_dim', None)
            num_heads = getattr(child, 'num_heads', None)
            batch_first = getattr(child, 'batch_first', False)
            dropout = getattr(child, 'dropout', 0.0)
            if embed_dim is not None and num_heads is not None:
                new = MuPMultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
                setattr(module, name, new)
        else:
            replace_attention_modules(child)


def replace_readout_with_mupreadout(model, readout_attr_names=('cls', 'head', 'fc_out', 'classifier')):
    """
    Replace final classifier/readout with MuPReadout (zero-init).
    Searches typical attribute names and replaces first match.
    """
    for attr in readout_attr_names:
        if hasattr(model, attr):
            old = getattr(model, attr)
            if isinstance(old, nn.Linear):
                new = MuPReadout(old.in_features, old.out_features, bias=old.bias is not None)
                setattr(model, attr, new)
                _logger.info(f"Replaced readout {attr} with MuPReadout")
                return True
    return False


# -------------------------
# Base-shape metadata save/restore (lightweight emulation)
# -------------------------
def save_base_meta(model, path):
    """
    Create a minimal description of the 'base' model describing:
      - architecture sizes (per layer embed dims / feature dims)
      - names of readout(s)
      - which modules are embeddings vs hidden vs readout
    Save to JSON so larger models can load and align.
    This is a heuristic; adjust per your ParticleTransformer internals.
    """
    meta = {}
    # store top-level useful config if present
    cfg = {}
    for k in ('embed_dims', 'pair_embed_dims', 'num_heads', 'num_layers'):
        if hasattr(model, k):
            cfg[k] = getattr(model, k)
    meta['cfg'] = cfg

    # record parameter classes by name (simple heuristic)
    meta['param_classes'] = {}
    for name, param in model.named_parameters():
        # heuristic: classifier/readout contains 'cls' or 'head' or 'classifier'
        if 'cls' in name or 'head' in name or 'classifier' in name:
            meta['param_classes'][name] = 'readout'
        elif 'embed' in name or 'embedding' in name or 'proj' in name and ('input' in name or 'embed' in name):
            meta['param_classes'][name] = 'embed'
        else:
            meta['param_classes'][name] = 'hidden'
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    _logger.info(f"Saved base meta to {path}")
    return meta


def load_base_meta(path):
    with open(path, 'r') as f:
        meta = json.load(f)
    return meta


def apply_base_meta_to_model(model, meta):
    """
    Use base meta to ensure the new model has the same parameter-class mapping.
    This is a heuristic check to help you detect mismatches.
    """
    mismatches = []
    for name, cls in meta['param_classes'].items():
        if name not in dict(model.named_parameters()):
            mismatches.append(name)
    if mismatches:
        _logger.warning("The target model is missing parameter names from base meta. "
                        "You may need to map names between versions. Missing examples: " + ", ".join(mismatches))
    else:
        _logger.info("All base meta parameter names found in target model (good).")
    # This function just checks; actual replacement must be performed
    return mismatches


# -------------------------
# High-level helper to create a MuP-ready model
# -------------------------
def make_mup_model(cfg, base_meta_path=None, save_base_meta_path=None):
    """
    1. build a ParticleTransformer(**cfg)
    2. replace linears with MuPLinear
    3. optionally replace attention modules with MuPMultiheadAttention
    4. replace final readout with MuPReadout
    5. if save_base_meta_path -> save minimal meta
    6. if base_meta_path -> load and check compatibility
    """
    model = ParticleTransformer(**cfg)

    # first, replace all Linear layers (feedforwards, projections) with MuPLinear
    replace_linears_with_mup(model)

    # try to replace attention blocks (best-effort)
    replace_attention_modules(model)

    # replace final classifier with MuPReadout if present
    replace_readout_with_mupreadout(model)

    if save_base_meta_path:
        save_base_meta(model, save_base_meta_path)

    if base_meta_path:
        meta = load_base_meta(base_meta_path)
        apply_base_meta_to_model(model, meta)

    return model