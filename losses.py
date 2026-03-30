import torch
import torch.nn as nn
import torch.nn.functional as F

def spatial_contrastive_loss(query_feat, memory_feat, query_labels, memory_labels, temperature=0.07):
    """
    query_feat:  (B, C, H, W)
    memory_feat: (N, C, H, W)
    """
    B, C, H, W = query_feat.shape
    N = memory_feat.shape[0]

    q = F.normalize(query_feat.mean(dim=[2,3]), dim=1)
    m = F.normalize(memory_feat.mean(dim=[2,3]), dim=1)

    sim = torch.matmul(q, m.T) / temperature

    pos_mask = (query_labels.unsqueeze(1) == memory_labels.unsqueeze(0))

    log_prob = F.log_softmax(sim, dim=1)
    loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
    return loss.mean()