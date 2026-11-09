"""Seed variance summary for Boltz-2, OpenFold3 and Protenix.
Same method as script 08: per complex, the coefficient of variation (std/mean)
across seeds for CDR-H3 RMSD and for ipTM; variance ratio = CV(RMSD) / CV(ipTM).
Written Nov 2026 to extend script 08 to OpenFold3 and Protenix, and to report the
median ratio and the ratio of mean CVs next to the mean of per-complex ratios.
Run from the repository root."""
import numpy as np
import pandas as pd

MODELS = {
    "Boltz-2":   ("data/boltz2_cdr_h3_rmsd.tsv",   "data/boltz2_analysis_ready.tsv",   "seed"),
    "OpenFold3": ("data/of3_cdr_h3_rmsd.tsv",      "data/of3_analysis_ready.tsv",      "seed_dir"),
    "Protenix":  ("data/protenix_cdr_h3_rmsd.tsv", "data/protenix_analysis_ready.tsv", "seed"),
}

def cv(x):
    return x.std() / x.mean() if x.mean() > 0 else np.nan

rows = []
for name, (rmsd_f, ana_f, seed_col) in MODELS.items():
    rmsd = pd.read_csv(rmsd_f, sep="\t")
    ana = pd.read_csv(ana_f, sep="\t")
    df = rmsd.merge(ana[["pdb", seed_col, "iptm"]], on=["pdb", seed_col], how="inner")
    g = df.groupby("pdb").agg(cv_cdr_h3=("cdr_h3_rmsd", cv), cv_iptm=("iptm", cv)).reset_index()
    g["variance_ratio"] = g["cv_cdr_h3"] / g["cv_iptm"]
    r = g["variance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    rows.append({
        "model": name,
        "predictions": len(df),
        "complexes": len(r),
        "mean_cv_cdr_h3": round(g["cv_cdr_h3"].mean(), 3),
        "mean_cv_iptm": round(g["cv_iptm"].mean(), 3),
        "mean_of_ratios": round(r.mean(), 2),
        "median_ratio": round(r.median(), 2),
        "ratio_of_mean_cvs": round(g["cv_cdr_h3"].mean() / g["cv_iptm"].mean(), 2),
        "n_ratio_gt1": int((r > 1).sum()),
        "pct_ratio_gt1": round(100 * (r > 1).mean(), 1),
        "max_ratio": round(r.max(), 1),
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("data/seed_variance_summary.tsv", sep="\t", index=False)
print("\nSaved: data/seed_variance_summary.tsv")
