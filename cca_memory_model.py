# code built off of https://github.com/dahyun-kang/renet/blob/main/models/renet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca import CCA 
from memorywrap import BatchMemoryWrapLayer
from torch.utils.checkpoint import checkpoint


class CCAMemoryModel(nn.Module):
    """
    backbone -> (feat_b, feat_m) -> CCA -> 2*B*N attn maps ->
    per-input attended memory [B,N,D] -> BatchMemoryWrapLayer -> logits.
    """

    def __init__(self, encoder, encoder_out_channels, num_classes, cca_reduce_dim=64, cca_hidden=16, temperature_attn=5.0, mem_wrap=None):
        super().__init__()
        self.encoder = encoder
        self.temperature_attn = temperature_attn

        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(encoder_out_channels, cca_reduce_dim, 1, bias=False),
            nn.BatchNorm2d(cca_reduce_dim),
            nn.ReLU(inplace=True),
        )
        # h(·) two-layer separable 4D conv (1 -> cca_hidden -> 1)
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[cca_hidden, 1])

        self.memory_wrap = mem_wrap or BatchMemoryWrapLayer(
            encoder_output_dim=encoder_out_channels,
            output_dim=num_classes,
        )

    
    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def _channel_mean_shift(self, x):
        return x - x.mean(1, keepdim=True)

    def _correlation_map(self, feat_b, feat_m):
        """
        feat_b: [B, C, Hb, Wb]
        feat_m: [N, C, Hm, Wm]
        return: [B, N, Hm, Wm, Hb, Wb]
        """
        fb = F.normalize(self.cca_1x1(feat_b), p=2, dim=1, eps=1e-8)
        fm = F.normalize(self.cca_1x1(feat_m), p=2, dim=1, eps=1e-8)
        # channel-wise cosine similarity at every (pos_m, pos_b) pair
        return torch.einsum('bchw,ncij->bnijhw', fb, fm)

    def compute_attention(self, feat_b, feat_m):
        feat_b = self._channel_mean_shift(feat_b)
        feat_m = self._channel_mean_shift(feat_m)

        corr = self._correlation_map(feat_b, feat_m)                  # [B,N,Hm,Wm,Hb,Wb]
        B, N, Hm, Wm, Hb, Wb = corr.shape

        corr = self.cca_module(corr.reshape(B * N, 1, Hm, Wm, Hb, Wb))
        corr = corr.reshape(B, N, Hm, Wm, Hb, Wb)

        cm = corr.reshape(B, N, Hm * Wm, Hb, Wb)
        cm = self.gaussian_normalize(cm, dim=2)
        cm = F.softmax(cm / self.temperature_attn, dim=2)
        cm = cm.reshape(B, N, Hm, Wm, Hb, Wb)
        attn_memory = cm.sum(dim=[4, 5])                              # [B, N, Hm, Wm]

        cb = corr.reshape(B, N, Hm, Wm, Hb * Wb)
        cb = self.gaussian_normalize(cb, dim=4)
        cb = F.softmax(cb / self.temperature_attn, dim=4)
        cb = cb.reshape(B, N, Hm, Wm, Hb, Wb)
        attn_batch = cb.sum(dim=[2, 3])                               # [B, N, Hb, Wb]

        return attn_batch, attn_memory                                # both [B, N, H, W]

    def forward(self, batch, memory, mem_chunk=100, return_extras=False):
        """
        batch:  [B, 3, H, W]
        memory: [N, 3, H, W]
        returns logits [B, num_classes] (+ optional dict of attn maps / embeddings)
        """
        feat_b = self.encoder(batch)
        feat_m = self.encoder(memory).detach()
        B, C, Hb, Wb = feat_b.shape
        N, _, Hm, Wm = feat_m.shape

        def process(fb, fm):
            attn_b, attn_m = self.compute_attention(fb, fm)
            att_mem = torch.einsum('bnhw,nchw->bnc', attn_m, fm) / (Hm * Wm)
            att_bat = torch.einsum('bnhw,bchw->bnc', attn_b, fb) / (Hb * Wb)
            return att_bat, att_mem, attn_b, attn_m

        att_bat_list, att_mem_list = [], []
        attn_b_list,  attn_m_list  = [], []
        for s in range(0, N, mem_chunk):
            fm_chunk = feat_m[s:s + mem_chunk]
            att_bat, att_mem, attn_b, attn_m = checkpoint(
                process, feat_b, fm_chunk, use_reentrant=False
            )
            att_bat_list.append(att_bat); att_mem_list.append(att_mem)
            if return_extras:
                attn_b_list.append(attn_b); attn_m_list.append(attn_m)

        attended_batch  = torch.cat(att_bat_list,  dim=1)    # [B, N, C]
        attended_memory = torch.cat(att_mem_list, dim=1)
        # Query into Memory-Wrap: the plain GAP of the batch features.
        encoder_output = feat_b.mean(dim=[-1, -2])                        # [B, C]
        out = self.memory_wrap(encoder_output, attended_memory, return_weights=return_extras)
        if return_extras:
            logits, memory_weights = out
        else:
            logits = out

        if return_extras:
            return logits, {
                'content_weights': memory_weights,   # [B, N]
                'attn_batch':      attn_batch,       # [B, N, Hb, Wb]
                'attn_memory':     attn_memory,      # [B, N, Hm, Wm]
                'attended_batch':  attended_batch,   # [B, N, C]
                'attended_memory': attended_memory,  # [B, N, C]
            }
        return logits