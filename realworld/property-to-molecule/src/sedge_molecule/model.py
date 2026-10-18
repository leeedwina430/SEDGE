"""Main CCOND architecture: an unconditional prefix, conditioned shared channels,
property-local updates, fusion blocks, and native token/ring heads."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from .policy import PROTOCOL

from .upstream import (
    RING_END_TOKENS,
    Transformer, TransformerConfig, TokenEmbedding, EdgeLogitLayer,
)


@dataclass(frozen=True)
class ModelConfig:
    """Explicit architecture settings, using the upstream ZINC GPT defaults.

    All settings are recorded in prepared assets and checkpoints."""
    emb_size: int = 1024
    num_layers: int = 3
    nhead: int = 16
    prefix_trunk_layers: int = 2          # blocks 0..1 are unconditional; block 2 performs fusion
    channel_c: int = 512
    channel_h1: int = 256
    channel_h2: int = 256
    max_len: int = 256
    n_correct: int = 20

    # Upstream architecture settings
    gpt: bool = True
    rotary: bool = True
    rmsnorm: bool = True
    swiglu: bool = True
    special_init: bool = True
    bias: bool = False                    # equivalent to --no_bias
    expand_scale: float = 2.0             # each SwiGLU projection has int(2.0*1024)=2048 units
    dropout: float = 0.0
    input_dropout: float = 0.0
    disable_treeloc: bool = False
    disable_graphmask: bool = False
    disable_valencemask: bool = False
    disable_counting_ring: bool = False
    enable_absloc: bool = False

    # Local conditioning modules
    local_cond_dim: int = 4
    local_emb_hidden: int = 128
    local_emb_out: int = 128
    local_update_hidden: int = 512
    local_rmsnorm_eps: float = 1e-6
    property_head_hidden: int = 256
    sigma_floor: float = 0.1
    sigma_cap: float = 5.0
    use_b_uncond: bool = True

    def __post_init__(self):
        if self.channel_c + self.channel_h1 + self.channel_h2 != self.emb_size:
            raise ValueError("Channel dimensions must sum to emb_size")
        if self.prefix_trunk_layers >= self.num_layers:
            raise ValueError("Prefix depth must leave at least one fusion layer")
        if self.enable_absloc and self.rotary:
            raise ValueError("Absolute and rotary positions cannot both be enabled")

    @property
    def ffn_hidden_per_projection(self) -> int:
        return int(self.expand_scale * self.emb_size)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ffn_hidden_per_projection"] = self.ffn_hidden_per_projection
        d["ffn_projections"] = 2 if self.swiglu else 1
        return d


class LocalModule(nn.Module):
    """Read (C, Hi) and the local property pack; return a residual update for Hi."""

    def __init__(self, cfg: ModelConfig, h_dim: int):
        super().__init__()
        self.norm_c = nn.RMSNorm(cfg.channel_c, eps=cfg.local_rmsnorm_eps, elementwise_affine=True)
        self.norm_h = nn.RMSNorm(h_dim, eps=cfg.local_rmsnorm_eps, elementwise_affine=True)
        # E_i: property pack -> 128
        self.cond_emb = nn.Sequential(
            nn.Linear(cfg.local_cond_dim, cfg.local_emb_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(cfg.local_emb_hidden, cfg.local_emb_out, bias=True),
        )
        in_dim = cfg.channel_c + h_dim + cfg.local_emb_out          # 512+256+128 = 896
        self.update = nn.Sequential(
            nn.Linear(in_dim, cfg.local_update_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(cfg.local_update_hidden, h_dim, bias=True),
        )
        # Zero initialization leaves the initial output independent of properties.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, c: torch.Tensor, h: torch.Tensor, pack: torch.Tensor) -> torch.Tensor:
        # pack: (B, 4) -> e: (B, 128) -> broadcast across every token in a molecule
        e = self.cond_emb(pack).unsqueeze(1).expand(-1, h.size(1), -1)
        v = torch.cat([self.norm_c(c), self.norm_h(h), e], dim=-1)
        return self.update(v)


class PropertyHead(nn.Module):
    """Predict (mu, sigma) from the prefix states before condition injection."""

    def __init__(self, cfg: ModelConfig, h_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.channel_c + h_dim, cfg.property_head_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(cfg.property_head_hidden, 2, bias=True),
        )
        self.sigma_floor = cfg.sigma_floor
        self.sigma_cap = cfg.sigma_cap
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias[0] = 0.0                                  # mu
            # sigma = floor + softplus(r) = 1.0  =>  r = log(expm1(1 - floor))
            final.bias[1] = math.log(math.expm1(1.0 - cfg.sigma_floor))

    def forward(self, c: torch.Tensor, h: torch.Tensor):
        out = self.net(torch.cat([c, h], dim=-1))
        mu = out[..., 0]
        sigma = (self.sigma_floor + nn.functional.softplus(out[..., 1])).clamp(max=self.sigma_cap)
        return mu, sigma


def route_local_mask(observed: torch.Tensor, side: int) -> torch.Tensor:
    """Expose only the property assigned to side 1 or 2."""
    keep = torch.zeros_like(observed)
    keep[:, side - 1] = True
    return observed & keep


def pack_interleaved(z: torch.Tensor, local_mask: torch.Tensor) -> torch.Tensor:
    """Pack [safe_z1, missing1, safe_z2, missing2], masking before multiplication."""
    safe = torch.where(local_mask, z, torch.zeros_like(z))
    miss = (~local_mask).to(z.dtype)
    return torch.stack([safe[:, 0], miss[:, 0], safe[:, 1], miss[:, 1]], dim=1)


class _GeneratorBase(nn.Module):
    def __init__(self, vocab, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.vocab = vocab
        c = self.cfg

        # Upstream components, initialized from scratch.
        self.token_embedding_layer = TokenEmbedding(len(vocab.TOKENS), c.emb_size)
        if not c.disable_counting_ring:
            self.count_embedding_layer = TokenEmbedding(c.max_len, c.emb_size)
        if not (c.enable_absloc or c.rotary):
            self.linear_loc_embedding_layer = nn.Embedding(c.max_len + 1, c.nhead)
        if not c.disable_treeloc:
            self.up_loc_embedding_layer = nn.Embedding(c.max_len + 1, c.nhead)
            self.down_loc_embedding_layer = nn.Embedding(c.max_len + 1, c.nhead)
        self.input_dropout = nn.Dropout(c.input_dropout)

        if not c.gpt:
            raise NotImplementedError("Only the upstream GPT path with separately accessible blocks is supported")
        self.transformer = Transformer(TransformerConfig(
            block_size=c.max_len, n_layer=c.num_layers, n_head=c.nhead, n_embd=c.emb_size,
            dropout=c.dropout, bias=c.bias, rotary=c.rotary, rmsnorm=c.rmsnorm,
            swiglu=c.swiglu, expand_scale=c.expand_scale,
        ))
        self.generator = nn.Linear(
            c.emb_size, len(vocab.TOKENS) - len(RING_END_TOKENS), bias=c.bias)
        self.ring_generator = EdgeLogitLayer(
            vocab=vocab, emb_size=c.emb_size, hidden_dim=c.emb_size,
            bias=c.bias, act=nn.SiLU() if c.swiglu else nn.GELU())

        # Apply upstream GPT-2 initialization before constructing local modules
        # to preserve their zero-initialized final layers.
        if c.special_init:
            self.apply(self._upstream_init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.num_layers))

        # Conditioning and auxiliary prediction components.
        # The bias-free trunk has no existing additive bias;
        # b_uncond supplies its learned additive constant.
        if c.use_b_uncond:
            self.b_uncond = nn.Parameter(torch.zeros(1, 1, c.emb_size))
        else:
            self.register_parameter("b_uncond", None)
        self.local1 = LocalModule(c, c.channel_h1)
        self.local2 = LocalModule(c, c.channel_h2)
        self.property_head1 = PropertyHead(c, c.channel_h1)
        self.property_head2 = PropertyHead(c, c.channel_h2)


    @staticmethod
    def _upstream_init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Attention mask matching upstream CondGenerator.forward.

    def _build_attn_mask(self, sequences, linear_loc, up_loc, down_loc):
        c = self.cfg
        b, L = sequences.size(0), sequences.size(1)
        mask = torch.zeros(b, L, L, c.nhead, device=sequences.device)
        if not (c.enable_absloc or c.rotary):
            mask = mask + self.linear_loc_embedding_layer(linear_loc)
        if not c.disable_treeloc:
            mask = mask + self.up_loc_embedding_layer(up_loc)
            mask = mask + self.down_loc_embedding_layer(down_loc)
        mask = mask.permute(0, 3, 1, 2)                              # b, nhead, L, L
        causal = (torch.triu(torch.ones((L, L), dtype=sequences.dtype,
                                        device=sequences.device)) == 1).transpose(0, 1)
        causal = causal.view(1, 1, L, L).expand(b, c.nhead, L, L)
        return mask.masked_fill(causal == 0, float("-inf"))

    # --- forward ----------------------------------------------------------

    def trunk(self, batched_mol_data):
        """Return unconditional (C, H1, H2, attention_mask) without reading targets."""
        (sequences, count_sequences, _graph_mask, _valence_mask,
         linear_loc, up_loc, down_loc) = batched_mol_data
        c = self.cfg
        x = self.token_embedding_layer(sequences)
        if not c.disable_counting_ring:
            x = x + self.count_embedding_layer(count_sequences)
        if self.b_uncond is not None:
            x = x + self.b_uncond
        x = self.input_dropout(x)
        attn = self._build_attn_mask(sequences, linear_loc, up_loc, down_loc)
        for block in self.transformer.transformer.h[: c.prefix_trunk_layers]:
            x = block(x, attn, None)
        i, j = c.channel_c, c.channel_c + c.channel_h1
        return x[..., :i], x[..., i:j], x[..., j:], attn


class SharedCConditioner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        spec=PROTOCOL['c_conditioner']
        self.norm_c=nn.RMSNorm(cfg.channel_c,eps=spec['norm_eps'],elementwise_affine=True)
        self.cond_emb=nn.Sequential(nn.Linear(4,spec['embedding_hidden']),nn.SiLU(),
                                    nn.Linear(spec['embedding_hidden'],spec['embedding_out']))
        self.update=nn.Sequential(nn.Linear(cfg.channel_c+spec['embedding_out'],spec['update_hidden']),
                                 nn.SiLU(),nn.Linear(spec['update_hidden'],cfg.channel_c))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self,c_base,z,observed):
        e=self.cond_emb(pack_interleaved(z,observed)).unsqueeze(1).expand(-1,c_base.size(1),-1)
        return self.update(torch.cat([self.norm_c(c_base),e],dim=-1))


def initial_conditioner(cfg):
    # New-module construction must not perturb the original model/dropout RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(PROTOCOL['c_conditioner']['init_seed'])
        return SharedCConditioner(cfg)


class ConditionedCGenerator(_GeneratorBase):
    def __init__(self,vocab):
        super().__init__(vocab,ModelConfig())
        self.c_condition=initial_conditioner(self.cfg)

    def _decode(self,batched_mol_data,z,observed,properties,return_states):
        sequences,_,graph_mask,valence_mask,_,_,_=batched_mol_data
        c_base,h1,h2,attn=self.trunk(batched_mol_data)
        delta=self.c_condition(c_base,z,observed)
        c_cond=c_base+delta
        m1=route_local_mask(observed,1)
        m2=route_local_mask(observed,2)
        v1=h1+self.local1(c_cond,h1,pack_interleaved(z,m1))
        v2=h2+self.local2(c_cond,h2,pack_interleaved(z,m2))
        fused=torch.cat([c_cond,v1,v2],dim=-1)
        for block in self.transformer.transformer.h[self.cfg.prefix_trunk_layers:]:
            fused=block(fused,attn,None)
        out=self.transformer.transformer.ln_f(fused)
        logits=torch.cat([self.generator(out),self.ring_generator(out,sequences)],dim=2)
        if not self.cfg.disable_graphmask:logits=logits.masked_fill(graph_mask,float('-inf'))
        if not self.cfg.disable_valencemask:logits=logits.masked_fill(valence_mask,float('-inf'))
        result={'logits':logits}
        if properties:
            # Keep the target out of both auxiliary heads. Do not detach c_base:
            # the original prefix-prediction task must continue training the trunk.
            mu1,sigma1=self.property_head1(c_base,h1)
            mu2,sigma2=self.property_head2(c_base,h2)
            result.update(property_mu=torch.stack([mu1,mu2],dim=-1),
                          property_sigma=torch.stack([sigma1,sigma2],dim=-1))
        if return_states:
            result['states']=dict(C=c_cond,C_base=c_base,C_cond=c_cond,C_delta=delta,
                                  H1=h1,H2=h2,V1=v1,V2=v2)
        return result

    def forward(self,batched_mol_data,z,observed,return_states=False):
        return self._decode(batched_mol_data,z,observed,True,return_states)

    def generation_logits(self,batched_mol_data,z,observed):
        if self.training or not bool(observed.all()):
            raise ValueError('Pure conditional sampling requires eval and full conditions')
        # Same core path as training, omitting only unused auxiliary predictions.
        return self._decode(batched_mol_data,z,observed,False,False)['logits'][:,-1].float()
