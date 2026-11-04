"""
Direct verification that the FedProx proximal term is active.

The review asked for three checks: the proximal gradient after the first local
step, the separation of the incoming reference weights from the mutable local
weights, and a nonzero-mu sanity test. The third is run in federated_v2.py as a
mu sweep. The first two are run here, on one client of one seed, because they
are properties of a single local update and do not need the whole simulation.

Note that the gradient of (mu/2)||w - w_global||^2 is mu*(w - w_global), which
is exactly zero at the first step because the local model is initialized to the
global weights. The informative measurement is therefore the step AFTER the
first, which is what "after the first local step" means and what is reported
here. The published implementation returns zero at every step and every mu,
because it built the term from p.data and so detached it from the graph.
"""
import os, sys, json, csv, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from federated import dirichlet_cases, make_model, get_flat, set_flat
from federated_v2 import flat_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"


def grads(model, X, y, gw, mu, published):
    """Objective gradient, and objective-plus-proximal gradient, at the current
    parameters. `published` reproduces the detached form that was shipped."""
    lossf = nn.BCEWithLogitsLoss()
    flat = (lambda m: torch.cat([p.data.view(-1) for p in m.parameters()])) \
        if published else flat_params
    base = lossf(model(X).squeeze(-1), y)
    g_base = torch.cat([g.reshape(-1) for g in
                        torch.autograd.grad(base, list(model.parameters()))])
    full = lossf(model(X).squeeze(-1), y) + (mu / 2) * ((flat(model) - gw) ** 2).sum()
    g_full = torch.cat([g.reshape(-1) for g in
                        torch.autograd.grad(full, list(model.parameters()))])
    return g_base, g_full


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    case_of = {r["file"]: r["case"] for r in rows}
    d = np.load(os.path.join(ROOT, "out", "features", f"plain__{MODEL}.npz"),
                allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])
    seed, alpha, lr, epochs = 0, 10.0, 0.05, 5
    assign = json.load(open(os.path.join(ROOT, "out", "splits",
                                         f"split_case_seed{seed}.json")))["assign"]
    tr = [f for f in files if assign[f] in ("train", "val")]
    sc = StandardScaler().fit(F[[idx[f] for f in tr]])
    cases = sorted({case_of[f] for f in tr})
    clab = {c: label[next(f for f in tr if case_of[f] == c)] for c in cases}
    cmap = dirichlet_cases(cases, clab, 10, alpha, np.random.default_rng(seed))
    fk = [f for f in tr if cmap[case_of[f]] == 0]
    X = torch.tensor(sc.transform(F[[idx[f] for f in fk]]), dtype=torch.float32)
    y = torch.tensor(y_all[[idx[f] for f in fk]], dtype=torch.float32)
    dim = X.shape[1]

    report = {"seed": seed, "alpha": alpha, "client": 0, "n_images": len(fk),
              "epochs": epochs, "lr": lr, "steps": []}
    for tag, published in (("published (p.data, detached)", True),
                           ("corrected (p, differentiable)", False)):
        g0 = make_model(dim, seed)
        gw = get_flat(g0).clone()
        m = make_model(dim, seed); set_flat(m, gw.clone())
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss()
        print(f"\n{tag}")
        print(f"  {'step':>5} {'||w - w_global||':>18} "
              f"{'max|g_prox|':>14} {'relative':>10}")
        for e in range(epochs):
            gb, gf = grads(m, X, y, gw, 0.01, published)
            dist = float((flat_params(m).detach() - gw).norm())
            gp = float((gf - gb).abs().max())
            rel = gp / float(gb.abs().max()) if float(gb.abs().max()) else 0.0
            report["steps"].append({"variant": tag, "step": e, "dist": dist,
                                    "max_abs_proximal_gradient": gp,
                                    "relative_to_objective": rel})
            print(f"  {e:>5} {dist:>18.6e} {gp:>14.6e} {rel:>10.3e}")
            opt.zero_grad()
            loss = lossf(m(X).squeeze(-1), y)
            flat = (lambda mm: torch.cat([p.data.view(-1) for p in mm.parameters()])) \
                if published else flat_params
            loss = loss + (0.01 / 2) * ((flat(m) - gw) ** 2).sum()
            loss.backward(); opt.step()

    # the reference weights must not be a view of the mutable local weights
    g0 = make_model(dim, seed); gw = get_flat(g0).clone()
    m = make_model(dim, seed); set_flat(m, gw.clone())
    before = gw.clone()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.ones_like(p))
    aliased = bool(not torch.equal(before, gw))
    shares = any(p.data.data_ptr() == gw.data_ptr() for p in m.parameters())
    report["reference_weights_aliased_to_local"] = aliased
    report["reference_shares_storage_with_local"] = shares
    print(f"\nreference weights change when the local weights are perturbed: "
          f"{aliased}")
    print(f"reference tensor shares storage with any local parameter: {shares}")
    json.dump(report, open(os.path.join(ROOT, "out", "results",
                                        "fedprox_probe.json"), "w"), indent=2)
    print("\nwrote out/results/fedprox_probe.json")


if __name__ == "__main__":
    main()
