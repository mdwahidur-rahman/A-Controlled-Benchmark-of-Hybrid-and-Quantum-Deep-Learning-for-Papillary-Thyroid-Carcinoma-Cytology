"""
Federated simulation over cached features, partitioned by CASE.

Honest framing, stated up front and repeated in the paper: this cohort is a
single hospital (Guangxi Medical University). There is no site structure to
recover, so this is a SIMULATED cross-silo federation, not a privacy
deployment. What is natural here is the grouping UNIT -- the patient case --
not the client boundary. Cases are assigned to clients under a Dirichlet prior
over the label distribution, so label skew is controlled and reproducible, and
no case is ever split across clients.

Strategies: FedAvg, FedProx (proximal term), SCAFFOLD (control variates),
against two reference points that bracket the achievable range:
  central     -- all data pooled. the upper bound.
  local_mean  -- each client alone, averaged. the no-collaboration floor.

Client-ensemble disagreement (std of client probabilities on the test set) is
recorded as the epistemic uncertainty signal, and its correlation with error is
reported -- the claim being tested is that disagreement is informative, not
merely present.
"""
import os, json, csv, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)


def dirichlet_cases(cases, labels, n_clients, alpha, rng):
    """Assign whole cases to clients with Dirichlet-controlled label skew."""
    assign = {}
    for cls in sorted(set(labels.values())):
        pool = [c for c in cases if labels[c] == cls]
        rng.shuffle(pool)
        p = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(pool)).astype(int)
        prev = 0
        for k, cut in enumerate(cuts):
            for c in pool[prev:cut]:
                assign[c] = k
            prev = cut
        for c in pool[prev:]:
            assign[c] = n_clients - 1
    return assign


def make_model(d, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, 1))


def get_flat(m):
    return torch.cat([p.data.view(-1) for p in m.parameters()])


def set_flat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


def local_train(model, X, y, epochs, lr, global_w=None, mu=0.0, c_local=None, c_global=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X).squeeze(-1)
        loss = lossf(out, y)
        if mu > 0 and global_w is not None:
            loss = loss + (mu / 2) * ((get_flat(model) - global_w) ** 2).sum()
        loss.backward()
        if c_local is not None:
            i = 0
            for p in model.parameters():
                n = p.numel()
                p.grad.data += (c_global[i:i + n] - c_local[i:i + n]).view_as(p)
                i += n
        opt.step()
    return model


def evaluate(model, X, y):
    with torch.no_grad():
        p = torch.sigmoid(model(X).squeeze(-1)).numpy()
    yh = (p >= 0.5).astype(int)
    return {"acc": float((yh == y.numpy()).mean()),
            "auc": float(roc_auc_score(y.numpy(), p))}, p


def run_strategy(name, clients, Xte, yte, d, rounds, epochs, lr, seed, mu=0.01):
    g = make_model(d, seed)
    gw = get_flat(g).clone()
    cg = torch.zeros_like(gw)
    cl = {k: torch.zeros_like(gw) for k in clients}
    ntot = sum(len(v[1]) for v in clients.values())
    hist = []
    for r in range(rounds):
        acc_w, new_cl = torch.zeros_like(gw), {}
        for k, (Xc, yc) in clients.items():
            m = make_model(d, seed); set_flat(m, gw.clone())
            if name == "fedprox":
                local_train(m, Xc, yc, epochs, lr, global_w=gw, mu=mu)
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
        gm = make_model(d, seed); set_flat(gm, gw)
        res, _ = evaluate(gm, Xte, yte)
        hist.append(res["acc"])
    gm = make_model(d, seed); set_flat(gm, gw)
    res, p = evaluate(gm, Xte, yte)
    return res, hist, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "out", "features"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "out", "splits"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "results"))
    ap.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 10.0])
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.manifest)))
    label = {r["file"]: int(r["label"]) for r in rows}
    case_of = {r["file"]: r["case"] for r in rows}
    d = np.load(os.path.join(a.features, f"{a.variant}__{a.model}.npz"), allow_pickle=True)
    F, files = d["features"], [str(x) for x in d["files"]]
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([label[f] for f in files])

    results = []
    for seed in a.seeds:
        sp = json.load(open(os.path.join(a.splits, f"split_case_seed{seed}.json")))
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
            clients, client_probs = {}, []
            for k in range(a.clients):
                fk = [f for f in tr if cmap[case_of[f]] == k]
                if len(fk) < 8 or len(set(y_all[[idx[f] for f in fk]])) < 2:
                    continue
                Xc = torch.tensor(sc.transform(F[[idx[f] for f in fk]]), dtype=torch.float32)
                yc = torch.tensor(y_all[[idx[f] for f in fk]], dtype=torch.float32)
                clients[k] = (Xc, yc)
            if len(clients) < 2:
                continue

            # no-collaboration floor + client ensemble for uncertainty
            local_accs = []
            for k, (Xc, yc) in clients.items():
                m = local_train(make_model(dim, seed), Xc, yc,
                                a.rounds * a.epochs, a.lr)
                r, p = evaluate(m, Xte, yte)
                local_accs.append(r["acc"]); client_probs.append(p)
            P = np.stack(client_probs)
            disagree = P.std(0)

            Xall = torch.tensor(sc.transform(F[[idx[f] for f in tr]]), dtype=torch.float32)
            yall = torch.tensor(y_all[[idx[f] for f in tr]], dtype=torch.float32)
            cm = local_train(make_model(dim, seed), Xall, yall, a.rounds * a.epochs, a.lr)
            cres, _ = evaluate(cm, Xte, yte)
            results.append({"strategy": "central", "alpha": alpha, "seed": seed,
                            "n_clients": len(clients), **cres})
            results.append({"strategy": "local_mean", "alpha": alpha, "seed": seed,
                            "n_clients": len(clients),
                            "acc": float(np.mean(local_accs)), "auc": float("nan")})

            for strat in ("fedavg", "fedprox", "scaffold"):
                res, hist, p = run_strategy(strat, clients, Xte, yte, dim,
                                            a.rounds, a.epochs, a.lr, seed)
                err = (np.abs(p - yte.numpy()) > 0.5).astype(float)
                corr = float(np.corrcoef(disagree, err)[0, 1]) if disagree.std() > 0 else float("nan")
                results.append({"strategy": strat, "alpha": alpha, "seed": seed,
                                "n_clients": len(clients),
                                "disagree_err_corr": corr, "history": hist, **res})

    with open(os.path.join(a.out, "federated.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"{'alpha':>6} {'strategy':12} {'acc':>14} {'auc':>8}  {'dis-err r':>9}")
    print("-" * 56)
    for alpha in a.alphas:
        for s in ("central", "fedavg", "fedprox", "scaffold", "local_mean"):
            v = [r["acc"] for r in results if r["strategy"] == s and r["alpha"] == alpha]
            au = [r["auc"] for r in results if r["strategy"] == s and r["alpha"] == alpha]
            cr = [r["disagree_err_corr"] for r in results
                  if r["strategy"] == s and r["alpha"] == alpha
                  and isinstance(r.get("disagree_err_corr"), float)
                  and r["disagree_err_corr"] == r["disagree_err_corr"]]
            if not v: continue
            cs = f"{np.mean(cr):9.3f}" if cr else "        -"
            print(f"{alpha:>6} {s:12} {np.mean(v)*100:7.2f}+/-{np.std(v)*100:4.2f} "
                  f"{np.nanmean(au):8.4f} {cs}")


if __name__ == "__main__":
    main()
