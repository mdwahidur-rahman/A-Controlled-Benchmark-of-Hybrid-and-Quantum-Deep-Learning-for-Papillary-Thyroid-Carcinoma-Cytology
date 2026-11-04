"""
Corrected federated simulation.

WHY THIS FILE EXISTS
--------------------
Three defects in federated.py are fixed here, and the first is not cosmetic.

1. THE PROXIMAL TERM HAD NO GRADIENT. federated.py formed it as

       loss = loss + (mu / 2) * ((get_flat(model) - global_w) ** 2).sum()

   where get_flat concatenates p.data, which is detached from the autograd
   graph. The term was therefore a constant with respect to the parameters and
   contributed exactly zero gradient at every step and every mu. FedProx was
   running FedAvg. The published statement that the trajectories coincided
   "as a property of the chosen mu" was wrong: they coincided because the
   proximal term was never applied. Here the term is built from the parameters
   themselves, and a mu sweep is run as a sanity test so that the dependence on
   mu is visible rather than asserted.

2. THE CENTRALIZED REFERENCE SAW MORE DATA THAN THE FEDERATION. Clients with
   fewer than eight fields or a single class are dropped, but the centralized
   model was fitted on the whole development partition including the dropped
   clients' data. A second reference, central_retained, is fitted on exactly
   the union of the retained clients, so that the federation is compared with a
   pooled model trained on the same data.

3. THE INVENTORY WAS NOT PUBLISHED. Retained and dropped clients, their image
   and group counts, their class composition and their aggregation weights
   n_k/n are now recorded per seed and per alpha.
"""
import os, json, csv, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from federated import dirichlet_cases, make_model, get_flat, set_flat, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)


def flat_params(m):
    """Differentiable flat view. The published version used p.data here, which
    detaches, and that is what silently disabled FedProx."""
    return torch.cat([p.view(-1) for p in m.parameters()])


def local_train(model, X, y, epochs, lr, global_w=None, mu=0.0,
                c_local=None, c_global=None, probe=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for e in range(epochs):
        opt.zero_grad()
        out = model(X).squeeze(-1)
        base = lossf(out, y)
        loss = base
        if mu > 0 and global_w is not None:
            loss = loss + (mu / 2) * ((flat_params(model) - global_w) ** 2).sum()
        loss.backward()
        if probe is not None and e == 0:
            # how much the proximal term actually moves the gradient, measured
            # at the first local step: the check the review asked for. The
            # objective-only gradient is recomputed on a fresh graph so that
            # neither backward pass disturbs the other.
            g_full = torch.cat([p.grad.view(-1) for p in model.parameters()]).clone()
            g_base = torch.cat(
                [g.reshape(-1) for g in torch.autograd.grad(
                    lossf(model(X).squeeze(-1), y), list(model.parameters()))]).clone()
            probe.append(float((g_full - g_base).abs().max()))
        if c_local is not None:
            i = 0
            for p in model.parameters():
                n = p.numel()
                p.grad.data += (c_global[i:i + n] - c_local[i:i + n]).view_as(p)
                i += n
        opt.step()
    return model


def run_strategy(name, clients, Xte, yte, d, rounds, epochs, lr, seed, mu=0.01,
                 probe=None):
    g = make_model(d, seed)
    gw = get_flat(g).clone()
    cg = torch.zeros_like(gw)
    cl = {k: torch.zeros_like(gw) for k in clients}
    ntot = sum(len(v[1]) for v in clients.values())
    hist, traj = [], []
    for r in range(rounds):
        acc_w, new_cl = torch.zeros_like(gw), {}
        for k, (Xc, yc) in clients.items():
            m = make_model(d, seed); set_flat(m, gw.clone())
            if name == "fedprox":
                local_train(m, Xc, yc, epochs, lr, global_w=gw.clone(), mu=mu,
                            probe=(probe if (r == 0 and probe is not None) else None))
            elif name == "scaffold":
                local_train(m, Xc, yc, epochs, lr, c_local=cl[k], c_global=cg)
                new_cl[k] = cl[k] - cg + (gw - get_flat(m)) / (epochs * lr)
            else:
                local_train(m, Xc, yc, epochs, lr)
            acc_w += get_flat(m) * (len(yc) / ntot)
        if name == "scaffold" and new_cl:
            cl = new_cl
            cg = torch.stack(list(cl.values())).mean(0)
        gw = acc_w
        traj.append(gw.clone())
        gm = make_model(d, seed); set_flat(gm, gw)
        res, _ = evaluate(gm, Xte, yte)
        hist.append(res["acc"])
    gm = make_model(d, seed); set_flat(gm, gw)
    res, p = evaluate(gm, Xte, yte)
    return res, hist, p, torch.stack(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 10.0])
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--mu-sweep", type=float, nargs="+",
                    default=[0.0, 0.001, 0.01, 0.1, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    RES = os.path.join(ROOT, "out", "results")

    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    label = {r["file"]: int(r["label"]) for r in rows}
    case_of = {r["file"]: r["case"] for r in rows}
    d = np.load(os.path.join(ROOT, "out", "features",
                             f"{a.variant}__{a.model}.npz"), allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])

    results, inventory, musweep, probes = [], [], [], []
    for seed in a.seeds:
        sp = json.load(open(os.path.join(ROOT, "out", "splits",
                                         f"split_case_seed{seed}.json")))
        assign = sp["assign"]
        tr = [f for f in files if assign[f] in ("train", "val")]
        te = [f for f in files if assign[f] == "test"]
        sc = StandardScaler().fit(F[[idx[f] for f in tr]])
        Xte = torch.tensor(sc.transform(F[[idx[f] for f in te]]), dtype=torch.float32)
        yte = torch.tensor(y_all[[idx[f] for f in te]], dtype=torch.float32)
        dim = Xte.shape[1]
        cases = sorted({case_of[f] for f in tr})
        clab = {c: label[next(f for f in tr if case_of[f] == c)] for c in cases}

        for alpha in a.alphas:
            rng = np.random.default_rng(seed)
            cmap = dirichlet_cases(cases, clab, a.clients, alpha, rng)
            clients, kept_files, inv = {}, [], []
            for k in range(a.clients):
                fk = [f for f in tr if cmap[case_of[f]] == k]
                yk = y_all[[idx[f] for f in fk]] if fk else np.array([])
                rec = {"seed": seed, "alpha": alpha, "client": k,
                       "n_images": len(fk),
                       "n_groups": len({case_of[f] for f in fk}),
                       "n_malignant": int((yk == 1).sum()),
                       "n_benign": int((yk == 0).sum())}
                if len(fk) < 8 or len(set(yk)) < 2:
                    rec["retained"] = False
                    rec["drop_reason"] = ("fewer than 8 fields" if len(fk) < 8
                                          else "single class")
                    rec["weight"] = 0.0
                    inv.append(rec); continue
                rec["retained"] = True; rec["drop_reason"] = None
                Xc = torch.tensor(sc.transform(F[[idx[f] for f in fk]]),
                                  dtype=torch.float32)
                yc = torch.tensor(yk, dtype=torch.float32)
                clients[k] = (Xc, yc); kept_files += fk; inv.append(rec)
            if len(clients) < 2:
                for rec in inv:
                    rec["round_used"] = False
                inventory += inv
                continue
            ntot = sum(len(v[1]) for v in clients.values())
            for rec in inv:
                rec["round_used"] = True
                rec["weight"] = (rec["n_images"] / ntot) if rec["retained"] else 0.0
            inventory += inv

            local_accs, client_probs = [], []
            for k, (Xc, yc) in clients.items():
                m = local_train(make_model(dim, seed), Xc, yc,
                                a.rounds * a.epochs, a.lr)
                r, p = evaluate(m, Xte, yte)
                local_accs.append(r["acc"]); client_probs.append(p)
            disagree = np.stack(client_probs).std(0)

            def fit_central(fset, tag):
                Xa = torch.tensor(sc.transform(F[[idx[f] for f in fset]]),
                                  dtype=torch.float32)
                ya = torch.tensor(y_all[[idx[f] for f in fset]], dtype=torch.float32)
                cm = local_train(make_model(dim, seed), Xa, ya,
                                 a.rounds * a.epochs, a.lr)
                cres, _ = evaluate(cm, Xte, yte)
                results.append({"strategy": tag, "alpha": alpha, "seed": seed,
                                "n_clients": len(clients),
                                "n_train_images": len(fset), **cres})
            fit_central(tr, "central")
            fit_central(kept_files, "central_retained")
            results.append({"strategy": "local_mean", "alpha": alpha, "seed": seed,
                            "n_clients": len(clients),
                            "n_train_images": len(kept_files),
                            "acc": float(np.mean(local_accs)), "auc": float("nan")})

            trajs = {}
            for strat in ("fedavg", "fedprox", "scaffold"):
                pr = [] if strat == "fedprox" else None
                res, hist, p, tj = run_strategy(strat, clients, Xte, yte, dim,
                                                a.rounds, a.epochs, a.lr, seed,
                                                mu=a.mu, probe=pr)
                trajs[strat] = tj
                err = (np.abs(p - yte.numpy()) > 0.5).astype(float)
                corr = (float(np.corrcoef(disagree, err)[0, 1])
                        if disagree.std() > 0 else float("nan"))
                results.append({"strategy": strat, "alpha": alpha, "seed": seed,
                                "n_clients": len(clients),
                                "n_train_images": len(kept_files),
                                "disagree_err_corr": corr, "history": hist, **res})
                if pr:
                    probes.append({"seed": seed, "alpha": alpha, "mu": a.mu,
                                   "max_abs_proximal_gradient": float(max(pr))})
            dmax = float((trajs["fedprox"] - trajs["fedavg"]).abs().max())
            results[-1]["fedprox_fedavg_max_param_gap"] = dmax

            if seed == a.seeds[0]:
                for mu in a.mu_sweep:
                    res, _, _, tj = run_strategy("fedprox", clients, Xte, yte, dim,
                                                 a.rounds, a.epochs, a.lr, seed, mu=mu)
                    musweep.append({"alpha": alpha, "seed": seed, "mu": mu,
                                    "acc": res["acc"], "auc": res["auc"],
                                    "max_param_gap_vs_fedavg":
                                    float((tj - trajs["fedavg"]).abs().max())})
            print(f"  seed {seed} alpha {alpha}: {len(clients)} clients retained, "
                  f"{len(kept_files)} of {len(tr)} images, "
                  f"FedProx-FedAvg max parameter gap {dmax:.3e}", flush=True)

    json.dump(results, open(os.path.join(RES, "federated_v2.json"), "w"), indent=2)
    json.dump({"inventory": inventory, "mu_sweep": musweep,
               "proximal_gradient_probe": probes},
              open(os.path.join(RES, "federated_inventory.json"), "w"), indent=2)

    print(f"\n{'alpha':>6} {'strategy':18} {'n':>3} {'acc %':>15} {'auc':>8}")
    print("-" * 56)
    for alpha in a.alphas:
        for s in ("central", "central_retained", "fedavg", "fedprox",
                  "scaffold", "local_mean"):
            v = [r["acc"] for r in results if r["strategy"] == s and r["alpha"] == alpha]
            au = [r["auc"] for r in results if r["strategy"] == s and r["alpha"] == alpha]
            if not v: continue
            print(f"{alpha:>6} {s:18} {len(v):>3} "
                  f"{np.mean(v)*100:7.2f} +/- {np.std(v)*100:4.2f} "
                  f"{np.nanmean(au):8.4f}")
    print("\nmu sweep (seed %d), FedProx:" % a.seeds[0])
    for r in musweep:
        print(f"  alpha {r['alpha']:>5}  mu {r['mu']:<6} acc {r['acc']*100:6.2f}  "
              f"max |w_prox - w_avg| {r['max_param_gap_vs_fedavg']:.4e}")
    print("\nproximal-gradient probe (max |grad with - grad without| at the first "
          "local step):")
    for r in probes:
        print(f"  seed {r['seed']} alpha {r['alpha']} mu {r['mu']}: "
              f"{r['max_abs_proximal_gradient']:.4e}")


if __name__ == "__main__":
    main()
