import argparse
import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from cox_baseline_patched import ACTUAL_CATEGORIES, bh_qvalues, build_survival_dataframe, fit_single_cox


def iter_dx_edges(categories: List[str], include_self_edges: bool) -> Iterable[Tuple[str, str]]:
    for a in categories:
        for b in categories:
            if not include_self_edges and a == b:
                continue
            yield a, b


def main():
    p = argparse.ArgumentParser(description="Run Cox baseline on all DX->DX edges using the same target-trial settings as training.")
    p.add_argument("--data", type=str, default="dataset_chronos.csv")
    p.add_argument("--out_dir", type=str, default="cox_results")
    p.add_argument("--seed", type=int, default=42)

    # mirror train.py defaults
    p.add_argument("--baseline_window", type=int, default=180)
    p.add_argument("--followup_window", type=int, default=365)
    p.add_argument("--washout_treatment", type=int, default=180)
    p.add_argument("--outcome_washout", type=int, default=30)
    p.add_argument("--buffer_days", type=int, default=14)
    p.add_argument("--active_obs_days", type=int, default=60)
    p.add_argument("--control_ratio", type=int, default=3)
    p.add_argument("--min_treated", type=int, default=50)

    p.add_argument("--fdr_alpha", type=float, default=0.05)
    p.add_argument("--ties", type=str, default="breslow", choices=["breslow", "efron"])
    p.add_argument("--include_self_edges", action="store_true")
    p.add_argument("--no_robust", action="store_true", help="Disable cluster-robust SE by patient id")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []
    for a, b in iter_dx_edges(ACTUAL_CATEGORIES, args.include_self_edges):
        print(f"\n>>>> COX DX->DX: A={a} | B={b} | d={args.buffer_days} | h={args.followup_window} <<<<")
        try:
            sdf = build_survival_dataframe(
                csv_path=args.data,
                treatment_codes=[a],
                outcome_codes=[b],
                baseline_window_days=args.baseline_window,
                followup_window_days=args.followup_window,
                washout_treatment_days=args.washout_treatment,
                outcome_washout_days=args.outcome_washout,
                buffer_days=args.buffer_days,
                active_obs_days=args.active_obs_days,
                control_ratio=args.control_ratio,
                min_treated=args.min_treated,
                seed=args.seed,
            )
            cohort_csv = os.path.join(args.out_dir, f"cox_cohort_DX_{a}_TO_{b}_d{args.buffer_days}_h{args.followup_window}.csv")
            sdf.to_csv(cohort_csv, index=False)

            summary, _ = fit_single_cox(sdf, ties=args.ties, robust=not args.no_robust)
            summary.update({
                "Malattia_A_(Exposure)": a,
                "Malattia_B_(Outcome)": b,
                "cohort_csv": cohort_csv,
            })
            summaries.append(summary)
            print(
                f"HR={summary['hazard_ratio']:.3f} "
                f"95%CI=[{summary['ci_lower_95']:.3f}, {summary['ci_upper_95']:.3f}] "
                f"p={summary['p_value']:.3g} n={summary['n']}"
            )
        except Exception as e:
            print(f"[WARN] Skipping {a}->{b}: {e}")
            summaries.append({
                "Malattia_A_(Exposure)": a,
                "Malattia_B_(Outcome)": b,
                "error": str(e),
            })

    out = pd.DataFrame(summaries)
    if "p_value" in out.columns:
        mask = out["p_value"].notna()
        if mask.any():
            out.loc[mask, "q_value"] = bh_qvalues(out.loc[mask, "p_value"].to_numpy())
        out["Significativo_FDR"] = np.where(out.get("q_value", 1.0) <= args.fdr_alpha, "Si", "No")

    sort_cols = [c for c in ["Significativo_FDR", "hazard_ratio", "p_value"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=[False, False, True][:len(sort_cols)])

    summary_csv = os.path.join(args.out_dir, "MASTER_COX_DX_DX.csv")
    out.to_csv(summary_csv, index=False)
    print(f"\n✅ Global Cox results saved to: {summary_csv}")


if __name__ == "__main__":
    main()
