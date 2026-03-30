import os
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def denormalize(tensor, mean, std):
    """Undo dataset normalisation for display. tensor: (3, H, W)"""
    t = tensor.clone().cpu().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_heatmap(img_np, map_np, orig_h, orig_w):
    """Resize map to image size and blend as jet heatmap."""
    m = torch.tensor(map_np)[None, None]
    m = F.interpolate(m, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    m = m.squeeze().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    heatmap = plt.cm.jet(m)[..., :3]
    return 0.5 * img_np + 0.5 * heatmap


def visualize_batch(x, mem_x, query_maps, memory_maps, att_weights, labels,
                    epoch, batch_idx, mean, std, save_dir='vis', n_examples=16, top_k=4):
    """
    x            : (B, 3, H, W)
    mem_x        : (N, 3, H, W)
    query_maps   : (B, N, Hf, Wf)
    memory_maps  : (B, N, Hf, Wf)
    att_weights  : (B, N)
    """
    B       = min(n_examples, x.shape[0])
    N_total = memory_maps.shape[1]
    orig_h, orig_w = x.shape[2], x.shape[3]

    # column layout: [query img | query heatmap | sep | mem1 img | mem1 mmap | mem1 qmap | sep | ...]
    col_widths  = [3, 3, 0.3]
    for i in range(top_k):
        col_widths += [3, 3, 3]
        if i < top_k - 1:
            col_widths += [0.3]

    n_cols = len(col_widths)
    fig = plt.figure(figsize=(sum(col_widths), 3 * B))
    gs  = gridspec.GridSpec(B, n_cols, figure=fig,
                            width_ratios=col_widths,
                            hspace=0.15, wspace=0.05)

    for b in range(B):
        weights = att_weights[b].detach().cpu()
        top_idx = weights.topk(min(top_k, N_total)).indices.numpy()
        img_q   = denormalize(x[b], mean, std)

        # ── query image ──────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[b, 0])
        ax.imshow(img_q)
        ax.set_ylabel(f'label {labels[b].item()}', fontsize=8)
        if b == 0:
            ax.set_title('Query', fontsize=8, fontweight='bold')
        ax.axis('off')

        # ── query heatmap (top-1 memory) ─────────────────────────────────────
        top1_mem  = top_idx[0]
        qmap_top1 = query_maps[b, top1_mem].detach().cpu().numpy()
        ax = fig.add_subplot(gs[b, 1])
        ax.imshow(overlay_heatmap(img_q, qmap_top1, orig_h, orig_w))
        if b == 0:
            ax.set_title('Query\ntop-1 activation', fontsize=8, fontweight='bold')
        ax.axis('off')

        # separator
        fig.add_subplot(gs[b, 2]).axis('off')

        # ── top-k memory columns ─────────────────────────────────────────────
        for plot_n, mem_n in enumerate(top_idx):
            w     = weights[mem_n].item()
            img_m = denormalize(mem_x[mem_n], mean, std)   # fixed: pass mean/std
            mmap  = memory_maps[b, mem_n].detach().cpu().numpy()
            qmap  = query_maps[b, mem_n].detach().cpu().numpy()

            # 3 fixed cols + 3 per memory block + 1 separator between blocks
            base_col = 3 + plot_n * (3 + 1)   # fixed: was * 4 but separator only between blocks

            ax = fig.add_subplot(gs[b, base_col])
            ax.imshow(img_m)
            for spine in ax.spines.values():
                spine.set_edgecolor('#aaaaaa')
                spine.set_linewidth(1.5)
            if b == 0:
                ax.set_title(f'Memory #{plot_n+1}\nattn {w:.3f}', fontsize=8, fontweight='bold')
            ax.axis('off')

            ax = fig.add_subplot(gs[b, base_col + 1])
            ax.imshow(overlay_heatmap(img_m, mmap, orig_h, orig_w))
            for spine in ax.spines.values():
                spine.set_edgecolor('#aaaaaa')
                spine.set_linewidth(1.5)
            if b == 0:
                ax.set_title('Mem\ncorr', fontsize=8)
            ax.axis('off')

            ax = fig.add_subplot(gs[b, base_col + 2])
            ax.imshow(overlay_heatmap(img_q, qmap, orig_h, orig_w))
            for spine in ax.spines.values():
                spine.set_edgecolor('#aaaaaa')
                spine.set_linewidth(1.5)
            if b == 0:
                ax.set_title('Query\nactivation', fontsize=8)
            ax.axis('off')

            # separator after each block except the last
            if plot_n < top_k - 1:
                fig.add_subplot(gs[b, base_col + 3]).axis('off')

    fig.suptitle(f'Epoch {epoch}  |  batch {batch_idx}', fontsize=10, y=1.02)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'epoch{epoch:03d}_batch{batch_idx:04d}.png')
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close(fig)