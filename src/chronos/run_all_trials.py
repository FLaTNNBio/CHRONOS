import argparse
import glob
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ACTUAL_CATEGORIES = [
    "INFE", "NEOP", "ENDO", "BLD", "MENT", "NERV", "CIRC", "RESP",
    "DIGE", "GEN", "SKIN", "MUSC", "CONG", "PERI", "INJ", "SUPP",
]

TARGET_DRUGS: Dict[str, str] = {
    "A02BC": "Proton Pump Inhibitors (A02BC)",
    "B01AC": "Antithrombotic agents (B01AC)",
    "C09AA": "ACE Inhibitors (C09AA)",
    "C10AA": "Statins (C10AA)",
    "N02BE": "Paracetamol (N02BE)",
}


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def launch(cmd: List[str], cwd: str):
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False, cwd=cwd)


def iter_dx_edges(categories: List[str], include_self_edges: bool) -> Iterable[Tuple[str, str]]:
    for a in categories:
        for b in categories:
            if not include_self_edges and a == b:
                continue
            yield a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_chronos.csv")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--followup", type=int, default=365)
    ap.add_argument("--buffer_days", type=int, default=14)
    ap.add_argument("--fdr_alpha", type=float, default=0.05)
    ap.add_argument("--min_treated", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--include_self_edges", action="store_true")
    ap.add_argument("--skip_drug_trials", action="store_true")
    args = ap.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for old in glob.glob(os.path.join(out_dir, "parziale_*.csv")):
        os.remove(old)

    for a, b in iter_dx_edges(ACTUAL_CATEGORIES, args.include_self_edges):
        partial_csv = os.path.join(out_dir, f"parziale_DX_{a}_TO_{b}_d{args.buffer_days}_h{args.followup}.csv")
        print(f"\n>>>> LAUNCHING DX->DX TRIAL: A={a} | B={b} | d={args.buffer_days} | h={args.followup} <<<<")
        cmd = [
            sys.executable, "train.py",
            "--data", data_path,
            "--treatment_codes", a,
            "--outcome_codes", b,
            "--buffer_days", str(args.buffer_days),
            "--followup_window", str(args.followup),
            "--out_path", partial_csv,
            "--epochs", str(args.epochs),
            "--n_folds", str(args.n_folds),
            "--batch_size", str(args.batch_size),
            "--min_treated", str(args.min_treated),
            "--use_mine",
            "--weighted_sampling",
        ]
        launch(cmd, script_dir)
        if not os.path.exists(partial_csv):
            print(f"⚠️ Trial {a}->{b} did not produce an output file.")

    if not args.skip_drug_trials:
        for code, name in TARGET_DRUGS.items():
            exposure = f"DRUG:{code}"
            for b in ACTUAL_CATEGORIES:
                partial_csv = os.path.join(out_dir, f"parziale_DRUG_{code}_TO_{b}_d{args.buffer_days}_h{args.followup}.csv")
                print(f"\n>>>> LAUNCHING DRUG->DX TRIAL: A={name} | B={b} | d={args.buffer_days} | h={args.followup} <<<<")
                cmd = [
                    sys.executable, "train.py",
                    "--data", data_path,
                    "--treatment_codes", exposure,
                    "--outcome_codes", b,
                    "--buffer_days", str(args.buffer_days),
                    "--followup_window", str(args.followup),
                    "--out_path", partial_csv,
                    "--epochs", str(args.epochs),
                    "--n_folds", str(args.n_folds),
                    "--batch_size", str(args.batch_size),
                    "--min_treated", str(args.min_treated),
                    "--use_mine",
                    "--weighted_sampling",
                ]
                launch(cmd, script_dir)
                if not os.path.exists(partial_csv):
                    print(f"⚠️ Trial {name}->{b} did not produce an output file.")

    all_paths = sorted(glob.glob(os.path.join(out_dir, "parziale_*_d*_h*.csv")))
    if not all_paths:
        print("❌ No valid partial trial outputs found.")
        return

    dfs = []
    for f in all_paths:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Errore lettura {f}: {e}")
    if not dfs:
        print("❌ No readable outputs.")
        return

    final_df = pd.concat(dfs, ignore_index=True)
    if "Malattia_A_(Exposure)" in final_df.columns:
        final_df["Exposure_readable"] = final_df["Malattia_A_(Exposure)"].apply(
            lambda x: TARGET_DRUGS.get(str(x).replace("DRUG:", ""), x) if str(x).startswith("DRUG:") else x
        )

    if "p_value" in final_df.columns:
        final_df["q_value"] = bh_qvalues(final_df["p_value"].values)
        final_df["Significativo_FDR"] = np.where(final_df["q_value"] <= args.fdr_alpha, "Si", "No")

    sort_cols = [c for c in ["Significativo_FDR", "Causal_Effect_ATE", "p_value"] if c in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(by=sort_cols, ascending=[False, False, True][: len(sort_cols)])

    master_csv = os.path.join(out_dir, "MASTER_ATE_Graph_FDR.csv")
    final_df.to_csv(master_csv, index=False)
    print(f"✅ Global results saved to: {master_csv}")


if __name__ == "__main__":
    main()
