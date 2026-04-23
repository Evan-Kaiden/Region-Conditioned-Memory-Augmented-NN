import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_out)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def _pair_distance(x, y, kind='cosine'):
    """
    x: [B, D]   query, one per batch item
    y: [B, N, D]  memory, per batch item
    returns [B, N]
    """
    if kind == 'cosine':
        xn = F.normalize(x, dim=-1, eps=1e-6)
        yn = F.normalize(y, dim=-1, eps=1e-6)
        return 1.0 - torch.einsum('bd,bnd->bn', xn, yn)
    if kind == 'l2':
        return (x.unsqueeze(1) - y).pow(2).sum(-1)
    if kind == 'dot':
        return -torch.einsum('bd,bnd->bn', x, y)
    raise ValueError(kind)


class BatchMemoryWrapLayer(nn.Module):
    def __init__(self, encoder_output_dim, output_dim, classifier=None, distance='cosine'):
        super().__init__()
        self.distance = distance
        self.classifier = classifier or MLP(
            encoder_output_dim * 2, encoder_output_dim * 4, output_dim
        )

    def forward(self, encoder_output, memory_set, return_weights=False):
        dist = _pair_distance(encoder_output, memory_set, self.distance)
        content_weights = sparsemax(-dist, dim=1)

        memory_vector = torch.einsum('bn,bnd->bd', content_weights, memory_set)

        out = self.classifier(torch.cat([encoder_output, memory_vector], 1))

        return (out, content_weights) if return_weights else out