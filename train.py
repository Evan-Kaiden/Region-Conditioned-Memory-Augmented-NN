import os
import torch
import torch.nn.functional as F

from generate_visuals import visualize_batch
from utils import get_mask_weight
from losses import spatial_contrastive_loss
from tqdm import tqdm 

def train_epoch(args, model, train_loader, mem_loader, optimizer, criterion, device, epoch, logger):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    mem_iter = iter(mem_loader)
    for x, y in tqdm(train_loader, leave=False):
        # model.mask_weight = get_mask_weight(epoch)

        try:
            mem_x, mem_y = next(mem_iter)
        except StopIteration:
            mem_iter = iter(mem_loader)
            mem_x, mem_y = next(mem_iter)

        x, y = x.to(device), y.to(device)
        mem_x, mem_y = mem_x.to(device), mem_y.to(device)

        B = x.shape[0]
        optimizer.zero_grad()

        # if args.use_correlation:
        #     (logits, _att), memory_maps, query_maps = model(x, mem_x, return_weights=True)
        #     cont_loss = spatial_contrastive_loss(
        #         model.last_query_feat,
        #         model.last_memory_feat,
        #         y, mem_y,
        #     )
        #     cls_loss = criterion(logits, y)
        #     loss = cls_loss + 0.25 * cont_loss
        # else:
        #     # BaselineMemory path — no correlation, no contrastive loss
        #     (logits, _att), memory_maps, query_maps = model(x, mem_x, return_weights=True)
        #     loss = criterion(logits, y)
        logits = model(x, mem_x)
        loss = criterion(logits, y)


        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += B

    train_loss = total_loss / len(train_loader)
    train_acc  = 100. * correct / total
    logger.info(f'  [train] epoch {epoch:03d}  loss {train_loss:.4f}  acc {train_acc:.2f}%')
    return train_loss, train_acc


def test_epoch(args, model, dset, test_loader, mem_loader, criterion, device, epoch, logger):
    model.eval()
    # model.mask_weight = get_mask_weight(epoch)
    total_loss, correct, total = 0.0, 0, 0

    mem_iter = iter(mem_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            try:
                mem_x, _ = next(mem_iter)
            except StopIteration:
                mem_iter = iter(mem_loader)
                mem_x, _ = next(mem_iter)

            x, y = x.to(device), y.to(device)
            mem_x = mem_x.to(device)
            B = x.shape[0]

            logits, extras = model(x, mem_x, return_extras=True)
            # (logits, att_weights), memory_maps, query_maps = model(x, mem_x, return_weights=True)
            loss = criterion(logits, y)
            total_loss += loss.item()
            correct += logits.argmax(1).eq(y).sum().item()
            total += B

    if args.generate_viz:
        n_vis = min(16, x.shape[0])
        vis_dir = os.path.join(args.run_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        visualize_batch(
            x=x[:n_vis],
            mem_x=mem_x,
            query_maps=extras['attn_batch'][:n_vis],
            memory_maps=extras['attn_memory'][:n_vis],
            att_weights=extras['content_weights'][:n_vis],
            labels=y[:n_vis],
            epoch=epoch,
            batch_idx=batch_idx,
            mean=dset.mean,
            std=dset.std,
            save_dir=vis_dir,
        )

    test_loss = total_loss / len(test_loader)
    test_acc  = 100. * correct / total
    logger.info(f'  [test]  epoch {epoch:03d}  loss {test_loss:.4f}  acc {test_acc:.2f}%')
    return test_loss, test_acc


def run(args, model, optimizer, criterion, scheduler, dataset, device, logger):
    best_acc = 0.0
    best_ckpt = os.path.join(args.run_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        logger.info(f'\n=== Epoch {epoch}/{args.epochs} ===')

        train_loss, train_acc = train_epoch(
            args, model, dataset.train_loader, dataset.mem_loader,
            optimizer, criterion, device, epoch, logger,
        )
        test_loss, test_acc = test_epoch(
            args, model, dataset, dataset.test_loader, dataset.mem_loader,
            criterion, device, epoch, logger,
        )

        if scheduler is not None:
            scheduler.step()

        logger.info(
            f'  summary  train loss {train_loss:.4f}  acc {train_acc:.2f}%'
            f' | test loss {test_loss:.4f}  acc {test_acc:.2f}%'
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    'epoch':      epoch,
                    'model':      model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'test_acc':   best_acc,
                    'args':       vars(args),
                },
                best_ckpt,
            )
            logger.info(f'  ** new best {best_acc:.2f}% — saved to {best_ckpt} **')

    logger.info(f'\nTraining complete. Best test acc: {best_acc:.2f}%')