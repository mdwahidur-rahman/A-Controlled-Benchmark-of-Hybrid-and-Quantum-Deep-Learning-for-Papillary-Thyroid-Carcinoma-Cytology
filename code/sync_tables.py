"""Rewrite the manuscript's data tables directly from the result archive.

The manuscript tables had been transcribed by hand, and had drifted from the
computed results: 67 cells of the full benchmark table and several cells of the
hybrid table disagreed with out/tables/*.csv, the whole ECE column by roughly a
factor of two. This script makes main.tex a view on the archive instead of a
copy of it, so the two cannot diverge again.

Run after any re-analysis:  python code/sync_tables.py --tex paper/main.tex
"""
import argparse, csv, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _first_dir(*candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def _first_file(*candidates):
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


# the working tree keeps tables under out/tables and the manuscript in a sibling
# paper/ directory; the released package keeps them under results/tables and
# ../manuscript. Resolve either without the caller having to care.
TAB = _first_dir(os.path.join(ROOT, "out", "tables"),
                 os.path.join(ROOT, "results", "tables"))
DEFAULT_TEX = _first_file(
    os.path.join(os.path.dirname(ROOT), "paper", "main.tex"),
    os.path.join(os.path.dirname(ROOT), "manuscript", "main.tex"),
    os.path.join(ROOT, "..", "manuscript", "main.tex"))


def read(name):
    with open(os.path.join(TAB, name), newline="") as fh:
        return list(csv.DictReader(fh))


def esc(s):
    return s.replace("%", r"\%").replace("&", r"\&")


def multirow(label, n=2):
    return r"\multirow{%d}{*}{%s}" % (n, label)


def pm(acc):
    """'91.27 ± 1.66' -> '91.27' (the sd lives in its own column elsewhere)."""
    return acc.split("±")[0].strip().replace("±", "")


def ci(text):
    """'[0.794, 0.855]' -> '0.794--0.855'. The bracketed form set the width of
    its column and pushed the table past the text block; the dashed range is
    three characters narrower and reads the same."""
    lo, hi = text.strip().strip("[]").split(",")
    return "%s--%s" % (lo.strip(), hi.strip())


def body_master1():
    rows = read("master1_full_benchmark.csv")
    by = {}
    for r in rows:
        by.setdefault(r["Architecture"], {})[r["Split"]] = r
    order = sorted(by, key=lambda k: -float(pm(by[k]["case"]["Accuracy %"])))
    best_acc = max(float(pm(by[k]["case"]["Accuracy %"])) for k in by)
    best_mcc = max(float(by[k]["case"]["MCC"]) for k in by)
    out = []
    for name in order:
        for i, sp in enumerate(("case", "image")):
            r = by[name][sp]
            acc, mcc = float(pm(r["Accuracy %"])), float(r["MCC"])
            a = r"\textbf{%s}" % pm(r["Accuracy %"]) if (sp == "case" and acc == best_acc) else pm(r["Accuracy %"])
            m = r"\textbf{%s}" % r["MCC"] if (sp == "case" and mcc == best_mcc) else r["MCC"]
            lead = multirow(esc(name)) if i == 0 else " " * 30
            # Error % = 100 - Accuracy % and kappa tracks MCC to three decimals
            # on this cohort, so both are dropped from the printed table and the
            # identity is stated in the caption. Keeping them forced the table
            # through \resizebox down to 6.1pt, below legible print size.
            out.append(
                f"{lead} & {sp:5} & {a} & {r['Sens %']} & {r['Spec %']} & "
                f"{r['FNR %']} & {r['FPR %']} & {r['Prec %']} & {r['F1 %']} & {m} & "
                f"{ci(r['MCC 95% CI'])} & {r['AUC']} & {r['ECE']} \\\\")
    return "\n".join(out)


def body_master2():
    rows = read("master2_hybrid_ablation.csv")
    by = {}
    for r in rows:
        by.setdefault(r["Configuration"], {})[r["Split"]] = r
    order = ["Swin-T alone", "ConvNeXtV2-T alone",
             "ConvNeXtV2-T, width-matched head (256)",
             "ConvNeXtV2-T, parameter-matched head (512)",
             "Concatenation, branch B shuffled (control)",
             "Concatenation, branch B deranged by case (control)",
             "Direct concatenation",
             "Concatenation + normalization + learned projection"]
    show = {"Concatenation + normalization + learned projection":
            "Concat + normalization + projection",
            "ConvNeXtV2-T, width-matched head (256)": "ConvNeXtV2-T, width-matched",
            "ConvNeXtV2-T, parameter-matched head (512)": "ConvNeXtV2-T, param-matched",
            "Concatenation, branch B shuffled (control)": "Concat, B shuffled",
            "Concatenation, branch B deranged by case (control)": "Concat, B deranged"}
    best = max(float(pm(by[k]["case"]["Accuracy %"])) for k in by)
    out = []
    for name in order:
        if name not in by:
            continue
        for i, sp in enumerate(("case", "image")):
            r = by[name][sp]
            acc = float(pm(r["Accuracy %"]))
            a = r"\textbf{%s}" % pm(r["Accuracy %"]) if (sp == "case" and acc == best) else pm(r["Accuracy %"])
            lead = multirow(esc(show.get(name, name))) if i == 0 else " " * 30
            # Error % is 100 - Accuracy %; dropped so the table prints at its
            # natural width instead of being shrunk to 6.9pt.
            out.append(
                f"{lead} & {sp:5} & {r['Dim']} & {a} & {r['Sens %']} & "
                f"{r['Spec %']} & {r['FNR %']} & {r['FPR %']} & {r['F1 %']} & {r['MCC']} & "
                f"{r['AUC']} \\\\")
    return "\n".join(out)


BUILDERS = {"tab:master1": body_master1, "tab:master2": body_master2}


def splice(tex, label, body):
    """Replace the rows between \\midrule and \\bottomrule of the table carrying `label`."""
    i = tex.index(r"\label{%s}" % label)
    j = tex.index(r"\midrule", i)
    k = tex.index(r"\bottomrule", j)
    return tex[:j] + "\\midrule\n" + body + "\n" + tex[k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=DEFAULT_TEX)
    ap.add_argument("--check", action="store_true", help="report drift without writing")
    a = ap.parse_args()
    tex = open(a.tex).read()
    new = tex
    for label, fn in BUILDERS.items():
        new = splice(new, label, fn())
    if a.check:
        print("DRIFT" if new != tex else "in sync")
        return 0 if new == tex else 1
    open(a.tex, "w").write(new)
    print(f"  synced {', '.join(BUILDERS)} in {a.tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
