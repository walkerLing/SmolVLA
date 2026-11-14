"""Parse SmolVLA train logs (20k + 80k runs) and plot training curves."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse(path):
    steps, loss, grdn, lr, mem = [], [], [], [], []
    pat = re.compile(r"step:(\d+(?:\.\d+)?)([KM]?) .*?loss:([\d.]+) grdn:([\d.]+) lr:([\deE.+-]+) .*?mem_gb:([\d.]+)")
    with open(path, errors="ignore") as f:
        for line in f.read().replace("\r", "\n").splitlines():
            m = pat.search(line)
            if m:
                s = float(m.group(1)) * (1000 if m.group(2) == "K" else 1e6 if m.group(2) == "M" else 1)
                steps.append(int(s)); loss.append(float(m.group(3)))
                grdn.append(float(m.group(4))); lr.append(float(m.group(5)))
                mem.append((int(s), float(m.group(6))))
    return steps, loss, grdn, lr, mem

M20 = "/root/autodl-tmp/outputs/_runmeta/libero_smolvla/train.log"
M80 = "/root/autodl-tmp/outputs/_runmeta/libero_smolvla_80k/train.log"
s1, l1, g1, r1, m1 = parse(M20)
s2, l2, g2, r2, m2 = parse(M80)  # 续训日志 step 为全局步数（80K 结尾），无需偏移
S, L, G, R = s1 + s2, l1 + l2, g1 + g2, r1 + r2

def smooth(y, w=20):
    return [sum(y[max(0, i - w):i + 1]) / len(y[max(0, i - w):i + 1]) for i in range(len(y))]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("SmolVLA LIBERO Training (batch=32, 20k run -> resume to 80k)", fontsize=13)

ax = axes[0][0]
ax.plot(S, L, alpha=0.25, color="tab:blue")
ax.plot(S, smooth(L), color="tab:blue", lw=2, label="loss (smoothed)")
ax.axvline(20000, color="red", ls="--", alpha=0.7)
ax.annotate("resume @20k\nLR jump 2.5e-6 -> 8.6e-5", xy=(20500, 1.2), fontsize=8, color="red")
ax.set(xlabel="global step", ylabel="flow-matching loss", title="Loss")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0][1]
ax.plot(S, R, color="tab:orange", lw=1.5)
ax.axvline(20000, color="red", ls="--", alpha=0.7)
ax.set(xlabel="global step", ylabel="learning rate", title="LR schedule (note the restart jump)", yscale="log")
ax.grid(alpha=0.3, which="both")

ax = axes[1][0]
ax.plot(S, G, alpha=0.2, color="tab:green")
ax.plot(S, smooth(G), color="tab:green", lw=2)
ax.axvline(20000, color="red", ls="--", alpha=0.7)
ax.set(xlabel="global step", ylabel="grad norm", title="Gradient norm (clip=10)")
ax.grid(alpha=0.3)

ax = axes[1][1]
ckpts = ["20k", "40k", "50k", "80k"]
spatial = [74.0, 57.0, 66.0, 65.0]
bars = ax.bar(ckpts, spatial, color=["tab:blue", "tab:gray", "tab:gray", "tab:red"], alpha=0.85)
for b, v in zip(bars, spatial):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=11)
ax.axhline(90, color="green", ls=":", label="paper (spatial) 90%")
ax.set(ylabel="libero_spatial pc_success (%)", ylim=(0, 100), title="Eval vs checkpoint (same seed=1000)")
ax.legend(); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = "/root/autodl-tmp/SmolVLA/training_curves.png"
plt.savefig(out, dpi=150)
print("saved:", out, "| points:", len(S), "| loss:", L[0], "->", round(L[-1], 3))
