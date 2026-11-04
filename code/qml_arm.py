"""
Quantum classification arm on cached cytology features.

Closes the gap left by quantum.py, which tested circuits in isolation on random
inputs. Here the circuits are trained as an actual transform block in the
classification pipeline, under the matched-capacity control bank from the
NeuroQFed-XAI scoping (§7).

Pipeline:  768-d features -> PCA(32, fit on train only) -> TRANSFORM -> linear head

Transform arms:
  identity          no transform block, 0 parameters          (control)
  dense_k           classical dense layer, matched parameters (control)
  vqc_amp_Y         amplitude encoding + all-Pauli-Y readout  (COLLAPSES)
  vqc_amp_mixed     amplitude encoding + mixed X/Y/Z readout  (ESCAPES)
  twin_amp_Y        the closed-form classical twin of vqc_amp_Y

The pre-registration rule applies: an arm that is exactly reproduced by its
twin cannot claim quantum advantage, however it scores. twin_amp_Y is included
precisely so the claim can be checked rather than asserted -- if it matches
vqc_amp_Y to numerical precision in ACCURACY as well as in expectation values,
the quantum layer is doing nothing a classical map could not.

Expectation, from the DR study: escaping the collapse did not buy accuracy
there (ring VQC 0.6371 vs identity 0.6982). Plan for the honest version of that
outcome here too.
"""
import os, sys, json, csv, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import all_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
N_QUBITS = 5
DIM = 2 ** N_QUBITS          # 32 amplitudes
I2 = torch.eye(2, dtype=torch.cfloat)
PX = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
PY = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat)
PZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)


def kron_at(op, j, n):
    M = torch.ones(1, 1, dtype=torch.cfloat)
    for q in range(n):
        M = torch.kron(M, op if q == j else I2)
    return M


def observables(pattern):
    m = {"X": PX, "Y": PY, "Z": PZ}
    return torch.stack([kron_at(m[pattern[j]], j, N_QUBITS) for j in range(N_QUBITS)])


class VQC(nn.Module):
    """Amplitude-encoded variational layer, state-vector simulated.

    The ansatz is built from per-qubit rotations plus a CNOT ring, repeated for
    `depth` layers -- the structure the audited papers describe as 'strongly
    entangling'. Simulation is exact (no shot noise), which is stated explicitly
    rather than implied: on an analytic simulator there is no measurement
    uncertainty, so nothing here is a quantum-derived uncertainty estimate.
    """
    def __init__(self, pattern="YYYYY", depth=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.rand(depth, N_QUBITS, 3, generator=g) * 2 * np.pi)
        self.register_buffer("obs", observables(pattern))
        self.depth = depth
        self.register_buffer("cnot", self._cnot_ring())

    def _cnot_ring(self):
        P = torch.eye(DIM, dtype=torch.cfloat)
        for j in range(N_QUBITS):
            k = (j + 1) % N_QUBITS
            M = torch.zeros(DIM, DIM, dtype=torch.cfloat)
            for b in range(DIM):
                bits = [(b >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS)]
                if bits[j] == 1:
                    bits[k] ^= 1
                bb = sum(v << (N_QUBITS - 1 - q) for q, v in enumerate(bits))
                M[bb, b] = 1.0
            P = M @ P
        return P

    def _rot(self, t):
        """Per-qubit Rz Ry Rz, tensored to the full space."""
        U = torch.ones(1, 1, dtype=torch.cfloat)
        for q in range(N_QUBITS):
            a, b, c = t[q]
            ca, sa = torch.cos(b / 2), torch.sin(b / 2)
            ry = torch.stack([torch.stack([ca + 0j, -sa + 0j]),
                              torch.stack([sa + 0j, ca + 0j])])
            rz1 = torch.diag(torch.stack([torch.exp(-1j * a / 2), torch.exp(1j * a / 2)]))
            rz2 = torch.diag(torch.stack([torch.exp(-1j * c / 2), torch.exp(1j * c / 2)]))
            U = torch.kron(U, rz2 @ ry @ rz1)
        return U

    def unitary(self):
        U = torch.eye(DIM, dtype=torch.cfloat)
        for d in range(self.depth):
            U = self.cnot @ self._rot(self.theta[d]) @ U
        return U

    def forward(self, x):
        psi = x.to(torch.cfloat)
        psi = psi / (psi.norm(dim=1, keepdim=True) + 1e-9)
        phi = psi @ self.unitary().T
        out = torch.einsum("bi,kij,bj->bk", phi.conj(), self.obs, phi)
        return out.real


class TwinAmpY(nn.Module):
    """Closed-form classical twin of the all-Y amplitude circuit.

    <Y_j> = sum_k s_jk * |(V U psi)_k|^2  -- one unitary, square magnitudes,
    fixed +/-1 readout. Shares the VQC's parameterization so the comparison is
    like for like.
    """
    def __init__(self, depth=2, seed=0):
        super().__init__()
        self.vqc = VQC("YYYYY", depth, seed)
        v = torch.tensor([[1, -1j], [1, 1j]], dtype=torch.cfloat) / np.sqrt(2)
        V = torch.ones(1, 1, dtype=torch.cfloat)
        for _ in range(N_QUBITS):
            V = torch.kron(V, v)
        self.register_buffer("V", V)
        S = torch.tensor([[1.0 if (k >> (N_QUBITS - 1 - j)) & 1 == 0 else -1.0
                           for k in range(DIM)] for j in range(N_QUBITS)])
        self.register_buffer("S", S)

    def forward(self, x):
        psi = x.to(torch.cfloat)
        psi = psi / (psi.norm(dim=1, keepdim=True) + 1e-9)
        phi = psi @ self.vqc.unitary().T
        p = (phi @ self.V.T).abs() ** 2
        return p @ self.S.T


class UnitNorm(nn.Module):
    """x -> x/||x||, the normalization amplitude encoding performs implicitly.

    The circuit arms normalize their input inside VQC.forward, while the
    identity and dense controls did not. Any gap between them therefore
    confounded the circuit with the loss of the radial component of the
    representation. This module makes the normalization a separate, testable
    stage so that the two can be told apart.
    """
    def forward(self, x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-9)


def build(arm, seed):
    if arm == "identity":
        return nn.Identity(), DIM
    if arm == "norm_only":
        return UnitNorm(), DIM
    if arm == "dense_k":
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(DIM, N_QUBITS), nn.Tanh()), N_QUBITS
    if arm == "dense_k_norm":
        torch.manual_seed(seed)
        return nn.Sequential(UnitNorm(), nn.Linear(DIM, N_QUBITS), nn.Tanh()), N_QUBITS
    if arm == "vqc_amp_Y":
        return VQC("YYYYY", 2, seed), N_QUBITS
    if arm == "vqc_amp_mixed":
        return VQC("XYZYX", 2, seed), N_QUBITS
    if arm == "twin_amp_Y":
        return TwinAmpY(2, seed), N_QUBITS
    raise ValueError(arm)


def param_counts(arm, seed=0):
    """Trainable parameters of the transform block, the classifier head and
    the total. The published table reported only the transform, which made a
    30-parameter circuit look comparable to a 165-parameter dense layer while
    the identity arm quietly carried a 33-parameter head against the others'
    six."""
    block, outdim = build(arm, seed)
    head = nn.Linear(outdim, 1)
    nb = sum(p.numel() for p in block.parameters() if p.requires_grad)
    nh = sum(p.numel() for p in head.parameters() if p.requires_grad)
    return {"transform": nb, "classifier": nh, "total": nb + nh, "out_dim": outdim}


def train_arm(arm, Xtr, ytr, Xva, yva, Xte, seed, epochs=260, lr=0.05):
    torch.manual_seed(seed)
    block, outdim = build(arm, seed)
    head = nn.Linear(outdim, 1)
    model = nn.Sequential(block, head)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = -1.0, None
    for e in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr).squeeze(-1), ytr)
        loss.backward(); opt.step()
        if e % 10 == 0 or e == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xva).squeeze(-1)).numpy()
            acc = ((pv >= 0.5) == yva.numpy()).mean()
            if acc > best:
                best = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte).squeeze(-1)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "results"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    d = np.load(os.path.join(ROOT, "out", "features",
                             f"{a.variant}__{a.model}.npz"), allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])

    arms = ["identity", "norm_only", "dense_k", "dense_k_norm",
            "vqc_amp_Y", "twin_amp_Y", "vqc_amp_mixed"]
    res = []
    pc = {arm: param_counts(arm) for arm in arms}
    print(f"{'arm':16} {'transform':>10} {'classifier':>11} {'total':>7} {'out dim':>8}")
    for arm in arms:
        q = pc[arm]
        print(f"{arm:16} {q['transform']:>10} {q['classifier']:>11} {q['total']:>7} {q['out_dim']:>8}")
    print()
    for regime in ("case", "image"):
        for seed in a.seeds:
            sp = json.load(open(os.path.join(ROOT, "out", "splits",
                                             f"split_{regime}_seed{seed}.json")))["assign"]
            sel = {s: np.array([idx[f] for f in files if sp[f] == s])
                   for s in ("train", "val", "test")}
            sc = StandardScaler().fit(F[sel["train"]])
            pca = PCA(n_components=DIM, random_state=seed).fit(sc.transform(F[sel["train"]]))
            X = {s: torch.tensor(pca.transform(sc.transform(F[sel[s]])),
                                 dtype=torch.float32) for s in sel}
            Y = {s: torch.tensor(y_all[sel[s]], dtype=torch.float32) for s in sel}
            for arm in arms:
                p = train_arm(arm, X["train"], Y["train"], X["val"], Y["val"], X["test"], seed)
                m = all_metrics(y_all[sel["test"]], p)
                m.update(arm=arm, regime=regime, seed=seed)
                res.append(m)
            print(f"  {regime} seed{seed} done", flush=True)

    json.dump(res, open(os.path.join(a.out, "qml_arm.json"), "w"), indent=2)
    json.dump(pc, open(os.path.join(a.out, "qml_params.json"), "w"), indent=2)
    print(f"\n{'arm':16} {'case acc':>15} {'case MCC':>9} {'image acc':>15}")
    print("-" * 60)
    for arm in arms:
        c = [r["acc"] for r in res if r["arm"] == arm and r["regime"] == "case"]
        cm = [r["mcc"] for r in res if r["arm"] == arm and r["regime"] == "case"]
        i = [r["acc"] for r in res if r["arm"] == arm and r["regime"] == "image"]
        print(f"{arm:16} {np.mean(c)*100:8.2f} ± {np.std(c)*100:4.2f} {np.mean(cm):9.4f} "
              f"{np.mean(i)*100:8.2f} ± {np.std(i)*100:4.2f}")
    cq = np.mean([r["acc"] for r in res if r["arm"] == "vqc_amp_Y" and r["regime"] == "case"])
    ct = np.mean([r["acc"] for r in res if r["arm"] == "twin_amp_Y" and r["regime"] == "case"])
    print(f"\nvqc_amp_Y vs its classical twin: {abs(cq-ct)*100:.3f} pp apart")


if __name__ == "__main__":
    main()
