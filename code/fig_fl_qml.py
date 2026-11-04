"""
Figure 3 -- federated arm and the quantum transform block with its twin.

Replaces a draft in which several elements described experiments this study
never ran. What is drawn here is what the code does:

  * FedAvg aggregates SAMPLE-WEIGHTED, sum_k (n_k/n) w_k, not (1/K) sum_k w_k.
  * The skew levels are alpha = 0.1, 0.5, 10.0.
  * Three strategies are run, not one: FedAvg, FedProx (mu=0.01), SCAFFOLD.
  * The sampling unit distributed to clients is the RECOVERED GROUP, not the image.
  * The ansatz is per-qubit Rz Ry Rz followed by a CNOT ring, depth L=2.
  * Readout is the Pauli-Y family, not Z.
  * Encoding is amplitude only; angle encoding appears solely in the collapse
    test of Section 3.5, never in the classification pipeline.
  * The classical twin contains NO qubit register. It is U(theta), then a
    magnitude squaring, then a fixed +/-1 sign matrix -- the content of
    Eq. (14). Drawing it with a register would destroy the argument.
  * There is no [f ; <O>] concatenation anywhere in the quantum arm.

The twin is drawn parallel to the circuit rather than after it, because the
result being reported is that the two tracks agree to 6.661e-16.

Authored at 42pc = 7.00in so \\includegraphics never rescales the type.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "out", "figures")
os.makedirs(FIG, exist_ok=True)

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
W42 = 7.00

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "savefig.bbox": None,
})

# soft fills matched to the ink colors
def tint(hexcol, a=0.14):
    r, g, b = [int(hexcol[i:i+2], 16)/255 for i in (1, 3, 5)]
    return (1-a + a*r, 1-a + a*g, 1-a + a*b)


def box(ax, x, y, w, h, fc="white", ec=INK2, lw=0.9, r=0.035, z=2, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def label(ax, x, y, s, size=7.6, weight="normal", color=INK, ha="center",
          va="center", z=5, style="normal"):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight,
                   color=color, zorder=z, style=style, linespacing=1.35)


def arrow(ax, p0, p1, color=INK2, lw=1.1, z=3, ls="-", head=5.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_width=2.1,head_length={head}",
                                 mutation_scale=1.0, lw=lw, color=color,
                                 zorder=z, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def panel(ax, x, y, w, h, title, accent):
    box(ax, x, y, w, h, fc="white", ec=GRID, lw=1.1, r=0.05, z=1)
    tab = box(ax, x + 0.012, y + h - 0.075, 0.30, 0.058, fc=tint(accent, 0.22),
              ec=accent, lw=0.9, r=0.03, z=3)
    label(ax, x + 0.022, y + h - 0.046, title, size=8.6, weight="bold",
          color=accent, ha="left", z=6)
    return tab


# =====================================================================
fig = plt.figure(figsize=(W42, 5.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ---------------------------------------------------------------- panel (a)
box(ax, 0.010, 0.560, 0.980, 0.432, fc="white", ec=GRID, lw=1.1, r=0.05, z=1)
box(ax, 0.022, 0.938, 0.268, 0.044, fc=tint(C["blue"], 0.22), ec=C["blue"], lw=0.9, r=0.025, z=3)
label(ax, 0.032, 0.960, "(a)  Simulated federation", size=8.4, weight="bold",
      color=C["blue"], ha="left", z=6)

CY = [0.868, 0.782, 0.640]
CCOL = [C["blue"], C["violet"], C["aqua"]]
CNAME = ["Client 1", "Client 2", "Client $K$"]
CSUB  = [r"cases $\mathcal{P}_1$", r"cases $\mathcal{P}_2$", r"cases $\mathcal{P}_K$"]
CW    = ["1", "2", "K"]

for yy, col, nm, sub, kk in zip(CY, CCOL, CNAME, CSUB, CW):
    box(ax, 0.030, yy - 0.027, 0.108, 0.054, fc=tint(col, 0.16), ec=col, lw=0.9)
    label(ax, 0.084, yy + 0.010, nm, size=7.6, weight="bold")
    label(ax, 0.084, yy - 0.012, sub, size=6.6, color=INK2)
    arrow(ax, (0.140, yy), (0.164, yy))
    box(ax, 0.166, yy - 0.027, 0.132, 0.054, fc=tint(C["yellow"], 0.17),
        ec=C["yellow"], lw=0.9)
    label(ax, 0.232, yy + 0.010, "Local update", size=7.4, weight="bold")
    label(ax, 0.232, yy - 0.012, rf"$E$ epochs $\to\ \mathbf{{w}}_{{{kk}}}^{{t+1}}$",
          size=6.6, color=INK2)
    arrow(ax, (0.300, yy), (0.374, yy), color=col, lw=1.2)

label(ax, 0.084, 0.712, r"$\cdots$", size=11, color=INK2)
label(ax, 0.232, 0.712, r"$\cdots$", size=11, color=INK2)

box(ax, 0.376, 0.598, 0.196, 0.316, fc=tint(C["blue"], 0.09), ec=C["blue"], lw=1.1)
label(ax, 0.474, 0.886, "Coordinator", size=8.2, weight="bold")
box(ax, 0.390, 0.740, 0.168, 0.082, fc="white", ec=GRID, lw=0.8)
label(ax, 0.474, 0.781,
      r"$\mathbf{w}^{t+1}=\sum_{k}\dfrac{n_k}{n}\,\mathbf{w}_k^{t+1}$", size=8.0)
label(ax, 0.474, 0.712, "sample-weighted by $n_k/n$", size=6.8,
      color=C["red"], weight="bold")
label(ax, 0.474, 0.676, r"FedProx: $+\frac{\mu}{2}\|\mathbf{w}-\mathbf{w}^t\|^2$",
      size=6.3, color=INK2)
label(ax, 0.474, 0.648, "SCAFFOLD: control", size=6.3, color=INK2)
label(ax, 0.474, 0.630, r"variates $\mathbf{c}_k$, $\mathbf{c}$", size=6.3, color=INK2)
label(ax, 0.474, 0.610, r"$T$ rounds", size=6.3, color=INK2)

for yy, col in zip(CY, CCOL):
    arrow(ax, (0.574, yy), (0.626, yy), color=col, lw=1.2, ls=(0, (2.4, 1.8)))
    box(ax, 0.628, yy - 0.024, 0.078, 0.048, fc="white", ec=col, lw=0.9)
    label(ax, 0.667, yy, r"$\mathbf{w}^{t+1}$", size=7.4)
label(ax, 0.667, 0.712, r"$\cdots$", size=11, color=INK2)
label(ax, 0.600, 0.916, "broadcast", size=6.7, color=INK2, style="italic")

# protocol -- kept inside the panel, two short lines per bullet
box(ax, 0.722, 0.590, 0.254, 0.330, fc="white", ec=INK2, lw=0.9, ls=(0, (3, 2)))
label(ax, 0.849, 0.898, "Protocol", size=7.9, weight="bold")
SET = [(r"unit: recovered $\bf{group}$, not image", None, None),
       (r"skew: Dirichlet $\alpha = 0.1,\ 0.5,\ 10.0$", None, None),
       ("strategies: FedAvg, FedProx", None, None),
       (r"($\mu\!=\!0.01$) and SCAFFOLD", None, None),
       (r"$K\!=\!10$ nominal; a client is", None, None),
       (r"dropped if $<\!8$ images or one class", None, None),
       ("references: centralized pool", None, None),
       ("and local-only (no collaboration)", None, None)]
BULLET = {0, 1, 2, 4, 6}
for j, (s1, bold, s3) in enumerate(SET):
    yy = 0.862 - j * 0.0335
    xx = 0.742 if j in BULLET else 0.752
    if j in BULLET:
        ax.plot([0.733], [yy], marker="o", ms=1.9, color=C["blue"], zorder=5)
    label(ax, xx, yy, s1, size=6.3, ha="left", color=INK2)

# ---------------------------------------------------------------- panel (b)
box(ax, 0.010, 0.010, 0.980, 0.528, fc="white", ec=GRID, lw=1.1, r=0.05, z=1)
box(ax, 0.022, 0.486, 0.520, 0.044, fc=tint(C["violet"], 0.22), ec=C["violet"],
    lw=0.9, r=0.025, z=3)
label(ax, 0.032, 0.508, "(b)  Quantum transform block and its closed-form twin",
      size=8.4, weight="bold", color=C["violet"], ha="left", z=6)

# shared front end
box(ax, 0.026, 0.168, 0.112, 0.178, fc=tint(C["aqua"], 0.13), ec=C["aqua"], lw=0.9)
label(ax, 0.082, 0.320, "Frozen encoder", size=7.3, weight="bold")
label(ax, 0.082, 0.292, r"$\mathbf{f}\in\mathbb{R}^{d_m}$", size=8.0)
label(ax, 0.082, 0.266, r"$d_m = 288\!-\!2048$", size=6.5, color=INK2)
label(ax, 0.082, 0.232, "standardize", size=6.9, weight="bold")
label(ax, 0.082, 0.210, "(train partition only)", size=6.3, color=INK2)
label(ax, 0.082, 0.186, r"PCA $\to$ 32", size=6.9, weight="bold")
arrow(ax, (0.140, 0.257), (0.164, 0.257))

box(ax, 0.166, 0.212, 0.084, 0.090, fc="white", ec=INK2, lw=0.9)
label(ax, 0.208, 0.282, r"$\mathbf{z}\in\mathbb{R}^{32}$", size=7.4)
label(ax, 0.208, 0.256, r"$\boldsymbol{\psi}=\mathbf{z}/\|\mathbf{z}\|_2$", size=7.4)
label(ax, 0.208, 0.230, "unit length", size=6.3, color=INK2)

ax.plot([0.254, 0.270], [0.257, 0.257], color=INK2, lw=1.1, zorder=3)
ax.plot([0.270, 0.270], [0.120, 0.388], color=INK2, lw=1.1, zorder=3)
arrow(ax, (0.270, 0.388), (0.292, 0.388))
arrow(ax, (0.270, 0.120), (0.292, 0.120))

# ---- upper track: circuit
box(ax, 0.294, 0.292, 0.492, 0.178, fc=tint(C["violet"], 0.08), ec=C["violet"], lw=0.9)
label(ax, 0.306, 0.448, "Variational circuit", size=7.6, weight="bold",
      color=C["violet"], ha="left")
label(ax, 0.318, 0.394, "amplitude", size=6.5, weight="bold")
label(ax, 0.318, 0.376, "encode", size=6.5, weight="bold")
label(ax, 0.318, 0.352, r"$n=5$", size=6.3, color=INK2)

wx0, wx1 = 0.388, 0.682
wy = np.linspace(0.322, 0.424, 5)
for yy in wy:
    ax.plot([wx0, wx1], [yy, yy], color=INK2, lw=0.7, zorder=3)
    label(ax, wx0 - 0.005, yy, r"$|0\rangle$", size=6.1, ha="right", color=INK2)
for yy in wy:
    box(ax, 0.396, yy - 0.0105, 0.068, 0.021, fc="white", ec=C["violet"],
        lw=0.75, r=0.010, z=4)
label(ax, 0.430, wy[2], r"$R_zR_yR_z$", size=6.1, z=6)
cx = [0.486, 0.506, 0.526, 0.546, 0.566]
for k in range(5):
    j = (k + 1) % 5
    ax.plot([cx[k], cx[k]], [wy[k], wy[j]], color=INK2, lw=0.7, zorder=4)
    ax.add_patch(Circle((cx[k], wy[k]), 0.0032, fc=INK2, ec=INK2, zorder=5))
    ax.add_patch(Circle((cx[k], wy[j]), 0.0056, fc="white", ec=INK2, lw=0.7, zorder=5))
label(ax, 0.526, 0.306, r"CNOT ring,  $L=2$ layers", size=6.3, color=INK2)
for yy in wy:
    box(ax, 0.634, yy - 0.0105, 0.044, 0.021, fc=tint(C["orange"], 0.18),
        ec=C["orange"], lw=0.75, r=0.010, z=4)
label(ax, 0.656, wy[2], r"$\langle Y\rangle$", size=6.3, z=6)
label(ax, 0.706, 0.306, r"Pauli-$Y$ readout", size=6.3, color=INK2)
arrow(ax, (0.684, 0.388), (0.704, 0.388))
label(ax, 0.740, 0.400, r"$\langle Y_1\rangle\ldots\langle Y_5\rangle$", size=7.0)
label(ax, 0.740, 0.372, "5 values", size=6.3, color=INK2)

# ---- lower track: twin
box(ax, 0.294, 0.040, 0.492, 0.150, fc=tint(C["aqua"], 0.08), ec=C["aqua"], lw=0.9)
label(ax, 0.306, 0.168, "Closed-form classical twin  —  no qubit register",
      size=7.6, weight="bold", color=C["aqua"], ha="left")
TW = [(0.306, r"$U(\theta)$", r"same $\theta$", 0.080),
      (0.396, r"$\mathbf{c}=VU\boldsymbol{\psi}$", "basis change", 0.100),
      (0.506, r"$p_k=|c_k|^2$", "magnitude", 0.098),
      (0.614, r"$\mathbf{S}\in\{\pm1\}^{5\times32}$", "sign matrix", 0.108)]
for xx, s1, s2, ww in TW:
    box(ax, xx, 0.092, ww, 0.052, fc="white", ec=C["aqua"], lw=0.8, r=0.018)
    label(ax, xx + ww/2, 0.127, s1, size=6.6)
    label(ax, xx + ww/2, 0.106, s2, size=6.1, color=INK2)
for a_, b_ in ((0.386, 0.396), (0.496, 0.506), (0.604, 0.614)):
    arrow(ax, (a_, 0.118), (b_, 0.118), color=C["aqua"], lw=0.9, head=4.0)
arrow(ax, (0.722, 0.118), (0.744, 0.118), color=C["aqua"], lw=0.9, head=4.0)
label(ax, 0.512, 0.062, r"Eq. (15):   $\langle O_j\rangle=\sum_k s_{jk}\,p_k$",
      size=6.6, color=INK2)

# ---- join
box(ax, 0.800, 0.404, 0.180, 0.046, fc="white", ec=C["red"], lw=0.9, r=0.018)
label(ax, 0.890, 0.436, "max deviation", size=6.4, weight="bold", color=C["red"])
label(ax, 0.890, 0.416, r"$6.661\times10^{-16}$", size=6.4, weight="bold",
      color=C["red"])
arrow(ax, (0.786, 0.388), (0.846, 0.330), color=C["violet"], lw=1.0)
arrow(ax, (0.760, 0.118), (0.846, 0.232), color=C["aqua"], lw=1.0)
box(ax, 0.850, 0.198, 0.128, 0.136, fc=tint(C["blue"], 0.10), ec=C["blue"], lw=0.9)
label(ax, 0.914, 0.302, "Linear head", size=7.2, weight="bold")
label(ax, 0.914, 0.272, r"$p(\mathrm{PTC}\mid x)$", size=7.2)
label(ax, 0.914, 0.242, "identical training,", size=6.1, color=INK2)
label(ax, 0.914, 0.224, "schedule, seeds", size=6.1, color=INK2)

fig.savefig(os.path.join(FIG, "fig20_fl_qml.pdf"))
fig.savefig(os.path.join(FIG, "fig20_fl_qml.png"))
print("wrote fig20_fl_qml.pdf / .png")
