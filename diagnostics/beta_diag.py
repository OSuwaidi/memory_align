"""Adaptive-beta telemetry for MAL_SGD (adaptive=True).

Trains ResNet-18-GN (repo recipe) on a CIFAR-10 subset (MPS), snapshots every per-tensor
adaptive beta each optimizer step, and analyzes the distribution + evolution across layers
(weights vs biases vs norm-gains), holistically, plus %time>0.9, %time<0.1, mean & max.
"""
import importlib.util
import sys
import time
import warnings

import numpy as np
import torch
from torch import nn
from torchvision import datasets
from torchvision.models import resnet18

warnings.filterwarnings("ignore", message=".*device is not CUDA.*")
DEVICE = "mps"
DATA = "/Users/omar/Python/datasets/CIFAR10"
OUT = "/private/tmp/claude-501/-Users-omar-Python-mem-align--claude-worktrees-mem-align-momentum-scaling-92ee6b/4e997545-6653-49f2-8c67-29714fdbd158/scratchpad"

# import the LIVE optimizer (not the stale worktree copy)
spec = importlib.util.spec_from_file_location("mal_live", "/Users/omar/Python/mem_align/mal_sgd.py")
mal = importlib.util.module_from_spec(spec); spec.loader.exec_module(mal)
MAL_SGD = mal.MAL_SGD

LR, WD, BS, STEPS, SEED = 0.1, 5e-4, 128, 800, 0
torch.manual_seed(SEED); torch.mps.manual_seed(SEED); np.random.seed(SEED)

# ---- data (subset, normalized, on device) ----
ds = datasets.CIFAR10(root=DATA, train=True, download=False)
x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
idx = torch.randperm(len(ds.targets))[:12800]
xtr = ((x[idx] - mean) / std).to(DEVICE)
ytr = torch.tensor(ds.targets)[idx].to(DEVICE)

# ---- model: ResNet-18 + GroupNorm, CIFAR stem (repo recipe) ----
model = resnet18(norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c))
model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 10)
model.to(DEVICE)

opt = MAL_SGD(model.parameters(), lr=LR, beta=0.9, weight_decay=WD, adaptive=True, nesterov=False)

# ---- map each optimizer param (in beta order) to name/type/size ----
id2name = {id(p): n for n, p in model.named_parameters()}
def kind(name, ndim):
    if ndim >= 2:
        return "weight"          # conv / linear weight matrices
    return "bias" if name.endswith("bias") else "norm_weight"  # GN affine gain
meta = [(id2name[id(p)], p.ndim, p.numel(), kind(id2name[id(p)], p.ndim))
        for group in opt.param_groups for p in group["params"]]
names = [m[0] for m in meta]; ndims = np.array([m[1] for m in meta])
numels = np.array([m[2] for m in meta]); kinds = np.array([m[3] for m in meta])
print(f"{len(meta)} param tensors | weights={sum(kinds=='weight')} "
      f"biases={sum(kinds=='bias')} norm_weights={sum(kinds=='norm_weight')}")

@torch.no_grad()
def snap():
    vals = [b.detach().float().reshape(()) for group in opt.param_groups for b in group["beta"]]
    return torch.stack(vals).cpu().numpy()

# ---- train, snapshot beta every step ----
g = torch.Generator().manual_seed(SEED + 1)
betas = []; losses = []; step = 0; t0 = time.time()
while step < STEPS:
    perm = torch.randperm(len(ytr), generator=g).to(DEVICE)
    for j in range(0, len(ytr) - BS + 1, BS):
        if step >= STEPS:
            break
        b = perm[j:j + BS]
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(xtr[b]), ytr[b])
        loss.backward(); opt.step()
        betas.append(snap()); losses.append(loss.item()); step += 1
        if step % 100 == 0:
            print(f"  step {step}/{STEPS}  loss={loss.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)

B = np.array(betas)              # (steps, n_tensors)
np.savez(f"{OUT}/beta_telemetry.npz", B=B, names=np.array(names), kinds=kinds,
         numels=numels, ndims=ndims, losses=np.array(losses))
print(f"recorded B shape={B.shape} in {(time.time()-t0)/60:.1f} min -> beta_telemetry.npz")
