"""Analyze + visualize adaptive-beta telemetry from beta_telemetry.npz."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-omar-Python-mem-align--claude-worktrees-mem-align-momentum-scaling-92ee6b/4e997545-6653-49f2-8c67-29714fdbd158/scratchpad"
d = np.load(f"{OUT}/beta_telemetry.npz", allow_pickle=True)
B, names, kinds, numels = d["B"], d["names"], d["kinds"], d["numels"]
S, M = B.shape
ss = slice(S // 2, S)                       # steady-state = last 50% of steps
COL = {"weight": "#2166ac", "bias": "#d6604d", "norm_weight": "#1b7837"}


def stats(a):
    a = a.ravel()
    return dict(mean=a.mean(), max=a.max(), min=a.min(),
                p_hi=100 * (a > 0.9).mean(), p_lo=100 * (a < 0.1).mean())


print(f"\n=== Adaptive-beta telemetry: {S} steps x {M} param tensors ===")
o = stats(B)
print(f"OVERALL (all steps):     mean={o['mean']:.3f}  max={o['max']:.3f}  min={o['min']:.3f}  "
      f"%>0.9={o['p_hi']:.1f}  %<0.1={o['p_lo']:.1f}")
s = stats(B[ss])
print(f"STEADY-STATE (last 50%): mean={s['mean']:.3f}  max={s['max']:.3f}  min={s['min']:.3f}  "
      f"%>0.9={s['p_hi']:.1f}  %<0.1={s['p_lo']:.1f}")
print("\nBy layer type (steady-state):")
for k in ("weight", "bias", "norm_weight"):
    mask = kinds == k
    if mask.any():
        st = stats(B[ss][:, mask])
        print(f"  {k:12s} (n={mask.sum():2d}): mean={st['mean']:.3f}  max={st['max']:.3f}  "
              f"%>0.9={st['p_hi']:5.1f}  %<0.1={st['p_lo']:5.1f}")

# per-layer steady-state mean & size trend
layer_mean = B[ss].mean(0)
order = np.argsort(-layer_mean)
print("\nTop-5 highest-beta layers (steady-state mean):")
for i in order[:5]:
    print(f"  {layer_mean[i]:.3f}  {names[i]}  (numel={numels[i]:,}, {kinds[i]})")
print("Bottom-5 lowest-beta layers:")
for i in order[-5:]:
    print(f"  {layer_mean[i]:.3f}  {names[i]}  (numel={numels[i]:,}, {kinds[i]})")
r = np.corrcoef(np.log10(numels.astype(float)), layer_mean)[0, 1]
print(f"\ncorr(mean beta, log10 numel) = {r:+.2f}  (positive => bigger layers run higher/steadier beta)")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.32, wspace=0.26)
steps = np.arange(S)

# (A) evolution: holistic mean + 10-90 band + per-type means
axA = fig.add_subplot(gs[0, 0])
lo, hi = np.percentile(B, 10, 1), np.percentile(B, 90, 1)
axA.fill_between(steps, lo, hi, color="gray", alpha=.25, label="10–90 pctile (tensors)")
axA.plot(steps, B.mean(1), color="k", lw=2, label="holistic mean")
for k in ("weight", "bias", "norm_weight"):
    mask = kinds == k
    if mask.any():
        axA.plot(steps, B[:, mask].mean(1), color=COL[k], lw=1.5, label=f"{k} mean")
axA.axhline(0.9, color="r", ls=":", lw=1); axA.axhline(0.1, color="b", ls=":", lw=1)
axA.set_title("A · adaptive β evolution over steps"); axA.set_xlabel("optimizer step")
axA.set_ylabel("β"); axA.set_ylim(0, 1.02); axA.legend(fontsize=7.5); axA.grid(alpha=.25)

# (B) steady-state distribution, overall + per type
axB = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, 1, 41)
axB.hist(B[ss].ravel(), bins=bins, color="gray", alpha=.5, density=True, label="all")
for k in ("weight", "bias", "norm_weight"):
    mask = kinds == k
    if mask.any():
        axB.hist(B[ss][:, mask].ravel(), bins=bins, histtype="step", lw=2,
                 color=COL[k], density=True, label=k)
axB.axvline(0.9, color="r", ls=":", lw=1); axB.axvline(0.1, color="b", ls=":", lw=1)
axB.set_title("B · steady-state β distribution (last 50%)"); axB.set_xlabel("β")
axB.set_ylabel("density"); axB.legend(fontsize=7.5); axB.grid(alpha=.25)

# (C) %-of-time bars
axC = fig.add_subplot(gs[0, 2])
cats = ["all", "weight", "bias", "norm_weight"]
hi_p = [stats(B[ss])["p_hi"]] + [stats(B[ss][:, kinds == k])["p_hi"] if (kinds == k).any() else 0 for k in cats[1:]]
lo_p = [stats(B[ss])["p_lo"]] + [stats(B[ss][:, kinds == k])["p_lo"] if (kinds == k).any() else 0 for k in cats[1:]]
xp = np.arange(len(cats)); w = .38
axC.bar(xp - w / 2, hi_p, w, color="#c1121f", label="% β > 0.9")
axC.bar(xp + w / 2, lo_p, w, color="#3a5a9c", label="% β < 0.1")
axC.set_xticks(xp); axC.set_xticklabels(cats, rotation=15, fontsize=8)
axC.set_title("C · % of time in high/low β regime (steady-state)"); axC.set_ylabel("% of observations")
axC.legend(fontsize=8); axC.grid(alpha=.25, axis="y")

# (D) layer x step heatmap (depth-ordered by registration)
axD = fig.add_subplot(gs[1, :2])
im = axD.imshow(B.T, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower",
                extent=[0, S, 0, M], interpolation="nearest")
axD.set_title("D · per-tensor β heatmap (rows = layers in depth order, bottom=input → top=output)")
axD.set_xlabel("optimizer step"); axD.set_ylabel("param-tensor index")
fig.colorbar(im, ax=axD, label="β", fraction=0.025, pad=0.01)

# (E) size trend: per-layer steady mean beta vs numel
axE = fig.add_subplot(gs[1, 2])
for k in ("weight", "bias", "norm_weight"):
    mask = kinds == k
    if mask.any():
        axE.scatter(numels[mask], layer_mean[mask], s=28, color=COL[k], alpha=.8, label=k, edgecolor="k", lw=.3)
axE.set_xscale("log"); axE.set_title(f"E · steady β vs layer size\ncorr(β, log numel)={r:+.2f}")
axE.set_xlabel("parameter count (log)"); axE.set_ylabel("steady-state mean β")
axE.set_ylim(0, 1.02); axE.legend(fontsize=7.5); axE.grid(alpha=.25)

fig.suptitle(f"MAL_SGD adaptive-β telemetry — ResNet-18-GN on CIFAR-10 ({S} steps, {M} tensors, lr=0.1, wd=5e-4)",
             fontsize=12, y=0.99)
fig.savefig(f"{OUT}/beta_telemetry.png", bbox_inches="tight", dpi=130)
print(f"\nsaved beta_telemetry.png")
