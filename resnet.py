'''ResNet in PyTorch.

Reference:
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from memorywrap import MemoryWrapLayer as EncoderMemoryWrapLayer
# from memorywrap import BaselineMemory as MemoryWrapLayer
from memorywrap import MemoryWrapLayer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,
                 initialize=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if initialize:
            import math
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class MemoryResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,
                 use_correlation=True, initialize=True, dataset=None):
        super(MemoryResNet, self).__init__()
        self.in_planes = 64
        self.use_correlation = use_correlation
        self.dataset = dataset
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.mw = MemoryWrapLayer(512 * block.expansion, num_classes)
 
        if initialize:
            self._initialize_weights()
 
    def _initialize_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def apply_correlation_mask(self, query_feat, memory_feat):
        B, C, H, W = query_feat.shape
        N = memory_feat.shape[0] // B
 
        def minmax(x):
            mn = x.min(dim=-1, keepdim=True).values
            mx = x.max(dim=-1, keepdim=True).values
            return (x - mn) / (mx - mn + 1e-8)
 
        with torch.no_grad():
            q = F.normalize(query_feat,  dim=1)
            m = F.normalize(memory_feat, dim=1)
 
            q    = q.view(B, 1, C, H * W).permute(0, 1, 3, 2)   # (B, 1, H*W, C)
            m    = m.view(B, N, C, H * W)                         # (B, N, C, H*W)
            corr = torch.matmul(q, m)                             # (B, N, H*W, H*W)
 
            memory_mask = minmax(corr.sum(dim=2))  # (B, N, H*W)
            query_mask  = minmax(corr.sum(dim=3))  # (B, N, H*W)
 
            memory_maps = memory_mask.view(B, N, H, W)
            query_maps  = query_mask.view(B, N, H, W)
 
        mask_weight          = getattr(self, 'mask_weight', 1.0)
        memory_mask_spatial  = memory_maps.unsqueeze(2)           # (B, N, 1, H, W)
        m_feat               = memory_feat.view(B, N, C, H, W)
 
        m_out = m_feat * (1 - mask_weight) + m_feat * memory_mask_spatial * mask_weight
        m_out = m_out.view(B * N, C, H, W)
 
        return m_out, memory_maps, query_maps
 
    def forward(self, x, ss, return_weights=False):
        B = x.size(0)
        N = ss.size(0)

        out    = F.relu(self.bn1(self.conv1(x)))
        out_mw = F.relu(self.bn1(self.conv1(ss)))

        out    = self.layer1(out);    out_mw = self.layer1(out_mw)
        out    = self.layer2(out);    out_mw = self.layer2(out_mw)
        out    = self.layer3(out);    out_mw = self.layer3(out_mw)

        if self.dataset in ('cifar10', 'cifar100'):
            if self.use_correlation:
                self.last_query_feat  = out
                self.last_memory_feat = out_mw
                out_mw_exp = (out_mw.unsqueeze(0)
                                    .expand(B, -1, -1, -1, -1)
                                    .reshape(B * N, *out_mw.shape[1:]))
                out_mw, memory_maps, query_maps = self.apply_correlation_mask(out, out_mw_exp)
            else:
                out_mw = (out_mw.unsqueeze(0)
                                .expand(B, -1, -1, -1, -1)
                                .reshape(B * N, *out_mw.shape[1:]))
                H, W        = out.shape[2], out.shape[3]
                memory_maps = torch.zeros(B, N, H, W, device=x.device)
                query_maps  = torch.zeros(B, N, H, W, device=x.device)

        out    = self.layer4(out);    out_mw = self.layer4(out_mw)

        if self.dataset in ('stl10', 'oxfordpets'):
            if self.use_correlation:
                self.last_query_feat  = out
                self.last_memory_feat = out_mw
                out_mw_exp = (out_mw.unsqueeze(0)
                                    .expand(B, -1, -1, -1, -1)
                                    .reshape(B * N, *out_mw.shape[1:]))
                out_mw, memory_maps, query_maps = self.apply_correlation_mask(out, out_mw_exp)
            else:
                out_mw = (out_mw.unsqueeze(0)
                                .expand(B, -1, -1, -1, -1)
                                .reshape(B * N, *out_mw.shape[1:]))
                H, W        = out.shape[2], out.shape[3]
                memory_maps = torch.zeros(B, N, H, W, device=x.device)
                query_maps  = torch.zeros(B, N, H, W, device=x.device)

        out    = F.adaptive_avg_pool2d(out,    1).view(B, -1)
        out_mw = F.adaptive_avg_pool2d(out_mw, 1).view(B, N, -1)

        if return_weights:
            logits_list, weights_list = [], []
            for b in range(B):
                logits_b, w_b = self.mw(out[b:b+1], out_mw[b], return_weights=True)
                logits_list.append(logits_b)
                weights_list.append(w_b)
            logits      = torch.cat(logits_list,  dim=0)
            att_weights = torch.cat(weights_list, dim=0)
            out_mw = (logits, att_weights)
        else:
            out_list = []
            for b in range(B):
                out_list.append(self.mw(out[b:b+1], out_mw[b], return_weights=False))
            out_mw = torch.cat(out_list, dim=0)

        return out_mw, memory_maps, query_maps

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def MemoryResNet18(dataset, use_correlation=True):
    return MemoryResNet(BasicBlock, [2, 2, 2, 2], use_correlation=use_correlation, dataset=dataset)
 
def MemoryResNet34(dataset, use_correlation=True):
    return MemoryResNet(BasicBlock, [3, 4, 6, 3], use_correlation=use_correlation, dataset=dataset)
 
def MemoryResNet50(dataset, use_correlation=True):
    return MemoryResNet(Bottleneck, [3, 4, 6, 3], use_correlation=use_correlation, dataset=dataset)
 
def MemoryResNet101(dataset, use_correlation=True):
    return MemoryResNet(Bottleneck, [3, 4, 23, 3], use_correlation=use_correlation, dataset=dataset)
 
def MemoryResNet152(dataset, use_correlation=True):
    return MemoryResNet(Bottleneck, [3, 8, 36, 3], use_correlation=use_correlation, dataset=dataset)