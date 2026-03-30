import torchvision.models as tv_models

def load_pretrained_imagenet(model, arch='resnet18'):
    pretrained = getattr(tv_models, arch)(pretrained=True)
    pretrained_state = pretrained.state_dict()
    model_state = model.state_dict()

    remapped = {k.replace('downsample', 'shortcut'): v 
                for k, v in pretrained_state.items()}

    matched = {
        k: v for k, v in remapped.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    skipped = [k for k in remapped if k not in matched]
    print(f'Loaded {len(matched)} layers, skipped {len(skipped)}: {skipped}')

    model_state.update(matched)
    model.load_state_dict(model_state)
    return model


def set_seed(seed):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_mask_weight(epoch, warmup_epochs=10):
    return min(1.0, epoch / warmup_epochs)