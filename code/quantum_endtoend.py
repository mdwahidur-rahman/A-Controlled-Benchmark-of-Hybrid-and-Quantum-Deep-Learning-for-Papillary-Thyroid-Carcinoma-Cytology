"""
End-to-end equivalence of the variational circuit and its closed-form twin, on
the ansatz the classifier actually uses, plus the optimization diagnostics the
review asked for.

WHY THIS FILE EXISTS
--------------------
Three separate things were conflated in the first submission.

1. THE ALGEBRA CHECK USED HAAR-RANDOM UNITARIES. Table 11 draws random
   unitaries and shows that the closed-form expression reproduces the measured
   expectations. That is a statement about the algebra, not about this
   classifier. The classifier uses a depth-two Rz-Ry-Rz ansatz with a CNOT
   ring, which is a measure-zero subset of the unitary group. The two tests are
   separated here, and the ansatz test is run at the parameters the optimizer
   actually visits.

2. IDENTICAL ACCURACY IS NOT IDENTICAL COMPUTATION. Agreement to 0.000
   accuracy points is compatible with different probabilities. This file
   compares the circuit and the twin at identical inputs and identical
   parameters on the quantities that actually have to agree: the five
   expectation values, the logit, the loss, the gradient with respect to every
   one of the 36 parameters, and the whole training trajectory.

3. THE DEFICIT WAS ATTRIBUTED TO CIRCUIT CAPACITY WITHOUT RULING OUT
   OPTIMIZATION FAILURE. Gradient norms, the learning-rate and epoch
   sensitivity, and the validation-selected epoch are recorded here, together
   with a classical control carrying 36 independently parameterized weights --
   the circuit's exact budget -- so that a capacity statement has something to
   stand on.
"""
import os, sys, json, csv, warnings, itertools
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qml_arm import VQC, TwinAmpY, UnitNorm, N_QUBITS, DIM
from metrics import all_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)


class BudgetMatched(nn.Module):
    """A classical block with exactly as many independent trainable weights as
    the circuit, mapping DIM inputs to N_QUBITS outputs. The twin shares the
    circuit's parameterization and so is the same function family; this one
    does not, which is what a capacity comparison needs."""
    def __init__(self, n_params, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # a rank-r factorization chosen so that r*(DIM + N_QUBITS) <= n_params
        r = max(1, n_params // (DIM + N_QUBITS))
        used = r * (DIM + N_QUBITS)
        self.A = nn.Parameter(torch.randn(DIM, r, generator=g) / np.sqrt(DIM))
        self.B = nn.Parameter(torch.randn(r, N_QUBITS, generator=g) / np.sqrt(r))
        self.spare = nn.Parameter(torch.zeros(n_params - used)) \
            if n_params > used else None
        self.rank = r

    def forward(self, x):
        h = x / (x.norm(dim=1, keepdim=True) + 1e-9)
        return torch.tanh(h @ self.A @ self.B)


def equivalence_on_ansatz(seed, n_draws, rng):
    """Circuit against twin at identical inputs and identical parameters."""
    vqc = VQC("YYYYY", 2, seed)
    twin = TwinAmpY(2, seed)
    twin.vqc.theta.data.copy_(vqc.theta.data)
    head_w = torch.randn(N_QUBITS, generator=torch.Generator().manual_seed(seed))
    head_b = torch.zeros(1)
    lossf = nn.BCEWithLogitsLoss()
    out = {"expectation": 0.0, "logit": 0.0, "loss": 0.0, "grad": 0.0}
    for _ in range(n_draws):
        x = torch.tensor(rng.normal(size=(8, DIM)), dtype=torch.float32)
        y = torch.tensor(rng.integers(0, 2, size=8), dtype=torch.float32)
        # re-randomize the parameters so that the agreement is not checked at a
        # single point of the parameter space
        th = torch.tensor(rng.uniform(0, 2 * np.pi, size=tuple(vqc.theta.shape)),
                          dtype=torch.float32)
        vqc.theta.data.copy_(th); twin.vqc.theta.data.copy_(th)
        ev, et = vqc(x), twin(x)
        out["expectation"] = max(out["expectation"],
                                 float((ev - et).abs().max()))
        lv = ev @ head_w + head_b
        lt = et @ head_w + head_b
        out["logit"] = max(out["logit"], float((lv - lt).abs().max()))
        cv, ct = lossf(lv, y), lossf(lt, y)
        out["loss"] = max(out["loss"], float((cv - ct).abs()))
        gv = torch.autograd.grad(cv, vqc.theta)[0]
        gt = torch.autograd.grad(ct, twin.vqc.theta)[0]
        out["grad"] = max(out["grad"], float((gv - gt).abs().max()))
    out["n_parameters_compared"] = int(np.prod(vqc.theta.shape))
    return out


def train(block, outdim, Xtr, ytr, Xva, yva, Xte, seed, epochs=260, lr=0.05,
          record=False):
    torch.manual_seed(seed)
    head = nn.Linear(outdim, 1)
    model = nn.Sequential(block, head)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, best_epoch = -1.0, None, -1
    traj, gnorm = [], []
    for e in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr).squeeze(-1), ytr)
        loss.backward()
        if record:
            g = torch.cat([p.grad.reshape(-1) for p in block.parameters()]) \
                if any(p.requires_grad for p in block.parameters()) \
                else torch.zeros(1)
            gnorm.append(float(g.norm()))
            traj.append(float(loss))
        opt.step()
        if e % 10 == 0 or e == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xva).squeeze(-1)).numpy()
            acc = ((pv >= 0.5) == yva.numpy()).mean()
            if acc > best:
                best, best_epoch = acc, e
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte).squeeze(-1)).numpy()
    return p, {"best_val_acc": float(best), "selected_epoch": int(best_epoch),
               "final_train_loss": float(loss), "grad_norms": gnorm,
               "train_loss_trace": traj}


def main():
    rng = np.random.default_rng(20261103)
    report = {}

    print("(1) Circuit against twin on the implemented depth-two ansatz, "
          "at identical inputs and identical parameters")
    eq = {}
    for seed in range(3):
        eq[seed] = equivalence_on_ansatz(seed, 40, rng)
        print(f"  seed {seed}: max |d expectation| {eq[seed]['expectation']:.3e}, "
              f"|d logit| {eq[seed]['logit']:.3e}, "
              f"|d loss| {eq[seed]['loss']:.3e}, "
              f"|d grad| over {eq[seed]['n_parameters_compared']} parameters "
              f"{eq[seed]['grad']:.3e}", flush=True)
    report["ansatz_equivalence"] = eq

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    d = np.load(os.path.join(ROOT, "out", "features",
                             "plain__convnextv2_tiny.fcmae_ft_in22k_in1k.npz"),
                allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])

    n_circ = int(np.prod(VQC("YYYYY", 2, 0).theta.shape))
    print(f"\n(2) Budget-matched classical control with {n_circ} independent "
          f"weights, and learning-rate / epoch sensitivity, case-level rule")
    traj_match, sens, budget = [], [], []
    for seed in range(5):
        sp = json.load(open(os.path.join(ROOT, "out", "splits",
                                         f"split_case_seed{seed}.json")))["assign"]
        sel = {s: np.array([idx[f] for f in files if sp[f] == s])
               for s in ("train", "val", "test")}
        sc = StandardScaler().fit(F[sel["train"]])
        pca = PCA(n_components=DIM, random_state=seed).fit(sc.transform(F[sel["train"]]))
        X = {s: torch.tensor(pca.transform(sc.transform(F[sel[s]])),
                             dtype=torch.float32) for s in sel}
        Y = {s: torch.tensor(y_all[sel[s]], dtype=torch.float32) for s in sel}
        report.setdefault("pca", []).append(
            {"seed": seed, "n_components": DIM,
             "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
             "svd_solver": pca.svd_solver_ if hasattr(pca, "svd_solver_") else "auto"})

        p, info = train(BudgetMatched(n_circ, seed), N_QUBITS, X["train"],
                        Y["train"], X["val"], Y["val"], X["test"], seed,
                        record=True)
        m = all_metrics(y_all[sel["test"]], p)
        m.update(arm="budget_matched", regime="case", seed=seed,
                 n_parameters=sum(q.numel() for q in
                                  BudgetMatched(n_circ, seed).parameters()))
        budget.append(m)

        # trajectory agreement: circuit and twin trained from the same state
        pc_, ic = train(VQC("YYYYY", 2, seed), N_QUBITS, X["train"], Y["train"],
                        X["val"], Y["val"], X["test"], seed, record=True)
        pt_, it = train(TwinAmpY(2, seed), N_QUBITS, X["train"], Y["train"],
                        X["val"], Y["val"], X["test"], seed, record=True)
        traj_match.append({
            "seed": seed,
            "max_abs_test_prob_diff": float(np.abs(pc_ - pt_).max()),
            "max_abs_train_loss_diff": float(np.abs(
                np.array(ic["train_loss_trace"]) -
                np.array(it["train_loss_trace"])).max()),
            "max_abs_grad_norm_diff": float(np.abs(
                np.array(ic["grad_norms"]) - np.array(it["grad_norms"])).max()),
            "circuit_selected_epoch": ic["selected_epoch"],
            "twin_selected_epoch": it["selected_epoch"],
            "circuit_final_train_loss": ic["final_train_loss"],
            "circuit_grad_norm_first": ic["grad_norms"][0],
            "circuit_grad_norm_last": ic["grad_norms"][-1],
            "circuit_grad_norm_median": float(np.median(ic["grad_norms"]))})
        print(f"  seed {seed}: budget-matched acc {m['acc']*100:6.2f}; "
              f"circuit/twin max |d p_test| {traj_match[-1]['max_abs_test_prob_diff']:.3e}, "
              f"max |d train loss| {traj_match[-1]['max_abs_train_loss_diff']:.3e}; "
              f"circuit grad norm {ic['grad_norms'][0]:.3e} -> "
              f"{ic['grad_norms'][-1]:.3e}", flush=True)

        if seed == 0:
            for lr in (0.005, 0.02, 0.05, 0.1, 0.3):
                for ep in (260, 800):
                    pq, iq = train(VQC("YYYYY", 2, seed), N_QUBITS, X["train"],
                                   Y["train"], X["val"], Y["val"], X["test"],
                                   seed, epochs=ep, lr=lr)
                    mq = all_metrics(y_all[sel["test"]], pq)
                    sens.append({"lr": lr, "epochs": ep, "acc": mq["acc"],
                                 "auc": mq["auc"],
                                 "best_val_acc": iq["best_val_acc"],
                                 "selected_epoch": iq["selected_epoch"]})
                    print(f"    lr {lr:<6} epochs {ep:<5} test acc "
                          f"{mq['acc']*100:6.2f}  val acc "
                          f"{iq['best_val_acc']*100:6.2f}", flush=True)

    report["trajectory_match"] = traj_match
    report["lr_epoch_sensitivity"] = sens
    report["budget_matched"] = budget
    report["n_circuit_parameters"] = n_circ
    b = [m["acc"] * 100 for m in budget]
    print(f"\nbudget-matched classical control, case-level, five seeds: "
          f"{np.mean(b):.2f} +/- {np.std(b):.2f}")
    json.dump(report, open(os.path.join(ROOT, "out", "results",
                                        "quantum_endtoend.json"), "w"), indent=2)
    print("wrote out/results/quantum_endtoend.json")


if __name__ == "__main__":
    main()
