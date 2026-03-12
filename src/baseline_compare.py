import argparse
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dataset import ChronosTargetTrialDataset

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# Representative edges already used in the paper.
PAPER_EDGES = [
    "BLD->ENDO",
    "NEOP->GEN",
    "MENT->MUSC",
    "ENDO->DIGE",
    "NEOP->DIGE",
    "INJ->DIGE",
    "MENT->NERV",
    "NERV->GEN",
]


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    m = p.size
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def parse_edges(mode: str, custom_edges: str = "") -> List[Tuple[str, str]]:
    if mode == "paper":
        raw = PAPER_EDGES
    elif mode == "custom":
        raw = [x.strip() for x in custom_edges.split(",") if x.strip()]
        if not raw:
            raise ValueError("--custom_edges is required when --mode custom")
    else:
        raise ValueError("Supported modes: paper, custom")

    edges = []
    for item in raw:
        if "->" not in item:
            raise ValueError(f"Invalid edge format: {item}. Use A->B")
        a, b = [x.strip().upper() for x in item.split("->", 1)]
        edges.append((a, b))
    return edges


def make_dataset(args, exposure: str, outcome: str) -> ChronosTargetTrialDataset:
    return ChronosTargetTrialDataset(
        csv_path=args.data,
        t0_trigger_codes=[exposure],
        treatment_codes=[exposure],
        outcome_codes=[outcome],
        baseline_window_days=args.baseline_window,
        followup_window_days=args.followup_window,
        washout_treatment_days=args.washout_treatment,
        outcome_washout_days=args.outcome_washout,
        buffer_days=args.buffer_days,
        active_obs_days=args.active_obs_days,
        control_ratio=args.control_ratio,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )


def dataset_to_tabular(ds: ChronosTargetTrialDataset) -> Dict[str, np.ndarray]:
    X, T, Y, Ypl, pids = [], [], [], [], []
    for i in range(len(ds)):
        sample = ds[i]
        seq_len = int(sample["seq_len"].item())
        target_idx = max(seq_len - 2, 0)
        cov = sample["current_covariates"][target_idx].numpy().astype(float)
        X.append(cov)
        T.append(float(sample["treated"].item()))
        # Single-outcome edge-specific dataset => take first component.
        Y.append(float(sample["outputs"][target_idx, 0].item()))
        Ypl.append(float(sample["placebo_outcome"][0].item()))
        pids.append(sample["pid"])
    return {
        "X": np.asarray(X, dtype=float),
        "T": np.asarray(T, dtype=float),
        "Y": np.asarray(Y, dtype=float),
        "Ypl": np.asarray(Ypl, dtype=float),
        "pid": np.asarray(pids),
        "n_treated": int(ds.n_treated),
        "n_control": int(ds.n_controls),
        "n": int(len(ds)),
    }


def make_folds(n: int, n_folds: int, seed: int) -> List[np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_folds = max(2, min(int(n_folds), n))
    return [fold.astype(int) for fold in np.array_split(idx, n_folds) if len(fold) > 0]


class ConstantProb:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-6, 1 - 1e-6))

    def predict_proba(self, X):
        p = np.full(len(X), self.p, dtype=float)
        return np.column_stack([1 - p, p])


def fit_logit(X: np.ndarray, y: np.ndarray):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return ConstantProb(float(np.mean(y)) if len(y) else 0.5)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y)
    return model


def crossfit_aipw_logistic(tab: Dict[str, np.ndarray], n_folds: int, seed: int, ps_eps: float) -> Dict[str, float]:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required for the logistic AIPW baseline.")

    X = tab["X"]
    T = tab["T"]
    Y = tab["Y"]
    Ypl = tab["Ypl"]
    folds = make_folds(len(X), n_folds, seed)

    mu0 = np.zeros(len(X), dtype=float)
    mu1 = np.zeros(len(X), dtype=float)
    ehat = np.zeros(len(X), dtype=float)

    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx, assume_unique=False)
        Xtr, Ttr, Ytr = X[train_idx], T[train_idx], Y[train_idx]

        ps_model = fit_logit(Xtr, Ttr)
        mu1_model = fit_logit(Xtr[Ttr == 1], Ytr[Ttr == 1]) if np.any(Ttr == 1) else ConstantProb(np.mean(Ytr))
        mu0_model = fit_logit(Xtr[Ttr == 0], Ytr[Ttr == 0]) if np.any(Ttr == 0) else ConstantProb(np.mean(Ytr))

        Xte = X[test_idx]
        ehat[test_idx] = ps_model.predict_proba(Xte)[:, 1]
        mu1[test_idx] = mu1_model.predict_proba(Xte)[:, 1]
        mu0[test_idx] = mu0_model.predict_proba(Xte)[:, 1]

    e_tilde = np.clip(ehat, ps_eps, 1.0 - ps_eps)
    trim_frac = float(np.mean((ehat < ps_eps) | (ehat > 1.0 - ps_eps)))

    psi = (mu1 - mu0) + (T / e_tilde) * (Y - mu1) - ((1.0 - T) / (1.0 - e_tilde)) * (Y - mu0)
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / math.sqrt(max(1, len(psi)))) if len(psi) > 1 else 0.0
    z = ate / se if se > 0 else 0.0
    p = float(2.0 * (1.0 - _norm_cdf(np.asarray([abs(z)]))[0])) if se > 0 else 1.0
    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se

    psi_pl = (T / e_tilde) * Ypl - ((1.0 - T) / (1.0 - e_tilde)) * Ypl
    placebo = float(np.mean(psi_pl))

    w = T / e_tilde + (1.0 - T) / (1.0 - e_tilde)
    ess = float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))

    return {
        "Causal_Effect_ATE": ate,
        "SE": se,
        "z_value": z,
        "p_value": p,
        "CI_Lower_95": float(ci_low),
        "CI_Upper_95": float(ci_high),
        "Placebo_ATE_IPW": placebo,
        "N": int(tab["n"]),
        "N_treated": int(tab["n_treated"]),
        "N_control": int(tab["n_control"]),
        "trim_frac": trim_frac,
        "ESS_ipw": ess,
        "e_min": float(np.min(ehat)),
        "e_median": float(np.median(ehat)),
        "e_max": float(np.max(ehat)),
        "n_folds": int(len(folds)),
        "cross_fitted": True,
    }


def naive_risk_difference(tab: Dict[str, np.ndarray]) -> Dict[str, float]:
    T = tab["T"]
    Y = tab["Y"]
    Ypl = tab["Ypl"]
    yt = Y[T == 1]
    yc = Y[T == 0]
    pt = Ypl[T == 1]
    pc = Ypl[T == 0]

    ate = float(yt.mean() - yc.mean())
    placebo = float(pt.mean() - pc.mean())
    var = float(np.var(yt, ddof=1) / max(1, len(yt)) + np.var(yc, ddof=1) / max(1, len(yc))) if len(yt) > 1 and len(yc) > 1 else 0.0
    se = math.sqrt(max(var, 0.0))
    z = ate / se if se > 0 else 0.0
    p = float(2.0 * (1.0 - _norm_cdf(np.asarray([abs(z)]))[0])) if se > 0 else 1.0
    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se

    return {
        "Causal_Effect_ATE": ate,
        "SE": float(se),
        "z_value": float(z),
        "p_value": p,
        "CI_Lower_95": float(ci_low),
        "CI_Upper_95": float(ci_high),
        "Placebo_ATE_IPW": placebo,
        "N": int(tab["n"]),
        "N_treated": int(tab["n_treated"]),
        "N_control": int(tab["n_control"]),
        "trim_frac": 0.0,
        "ESS_ipw": float(tab["n"]),
        "e_min": np.nan,
        "e_median": np.nan,
        "e_max": np.nan,
        "n_folds": 1,
        "cross_fitted": False,
    }


def run_train_variant(train_py: Path, args, exposure: str, outcome: str, method_name: str, out_dir: Path, extra_flags: List[str]) -> pd.DataFrame:
    out_csv = out_dir / f"{method_name}__{exposure}__{outcome}.csv"
    cmd = [
        os.environ.get("PYTHON", "python"),
        str(train_py),
        "--data", args.data,
        "--seed", str(args.seed),
        "--out_path", str(out_csv),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_seq_len", str(args.max_seq_len),
        "--latent_dim", str(args.latent_dim),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--n_folds", str(args.n_folds),
        "--min_treated", str(args.min_treated),
        "--t0_codes", exposure,
        "--treatment_codes", exposure,
        "--outcome_codes", outcome,
        "--baseline_window", str(args.baseline_window),
        "--followup_window", str(args.followup_window),
        "--washout_treatment", str(args.washout_treatment),
        "--outcome_washout", str(args.outcome_washout),
        "--buffer_days", str(args.buffer_days),
        "--active_obs_days", str(args.active_obs_days),
        "--control_ratio", str(args.control_ratio),
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--gamma", str(args.gamma),
        "--lambda_prop", str(args.lambda_prop),
        "--perc", str(args.perc),
        "--margin", str(args.margin),
        "--max_pairs", str(args.max_pairs),
        "--ps_eps", str(args.ps_eps),
        "--add_qvalues",
        "--weighted_sampling",
    ] + extra_flags

    print("\n>>> Running", method_name, "for", f"{exposure}->{outcome}")
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)
    df = pd.read_csv(out_csv)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        rows.append({
            "method": method,
            "edges": int(len(g)),
            "fdr_significant_edges": int(g["global_fdr_significant"].sum()) if "global_fdr_significant" in g.columns else int(g["fdr_significant"].sum()),
            "mean_ate": float(g["Causal_Effect_ATE"].mean()),
            "median_abs_placebo": float(g["Placebo_ATE_IPW"].abs().median()),
            "median_ess": float(g["ESS_ipw"].median()),
            "median_trim_frac": float(g["trim_frac"].median()),
        })
    return pd.DataFrame(rows).sort_values("method")


def main():
    p = argparse.ArgumentParser(description="Compare CHRONOS against simpler baselines on the same edge-level target-trial protocol.")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="baseline_compare")
    p.add_argument("--mode", type=str, default="paper", choices=["paper", "custom"])
    p.add_argument("--custom_edges", type=str, default="")
    p.add_argument("--methods", type=str, default="naive_rd,aipw_logistic,chronos_full,pred_only")
    p.add_argument("--train_py", type=str, default="train.py")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--ps_eps", type=float, default=0.01)

    # trial-design args
    p.add_argument("--min_treated", type=int, default=50)
    p.add_argument("--baseline_window", type=int, default=180)
    p.add_argument("--followup_window", type=int, default=365)
    p.add_argument("--washout_treatment", type=int, default=180)
    p.add_argument("--outcome_washout", type=int, default=30)
    p.add_argument("--buffer_days", type=int, default=14)
    p.add_argument("--active_obs_days", type=int, default=60)
    p.add_argument("--control_ratio", type=int, default=3)

    # train.py args
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_seq_len", type=int, default=60)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--lambda_prop", type=float, default=1.0)
    p.add_argument("--perc", type=float, default=30.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--max_pairs", type=int, default=2048)

    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_py = Path(args.train_py)
    edges = parse_edges(args.mode, args.custom_edges)

    edge_rows: List[Dict] = []

    for exposure, outcome in edges:
        print("\n" + "=" * 88)
        print(f"EDGE {exposure}->{outcome}")
        print("=" * 88)
        ds = make_dataset(args, exposure, outcome)
        if ds.n_treated < args.min_treated or len(ds) == 0:
            print(f"Skipping {exposure}->{outcome}: treated={ds.n_treated}, cohort={len(ds)}")
            continue
        tab = dataset_to_tabular(ds)

        for method in methods:
            if method == "naive_rd":
                res = naive_risk_difference(tab)
                row = {"method": method, "Malattia_A_(Exposure)": exposure, "Malattia_B_(Outcome)": outcome, **res}
                edge_rows.append(row)
            elif method == "aipw_logistic":
                res = crossfit_aipw_logistic(tab, args.n_folds, args.seed, args.ps_eps)
                row = {"method": method, "Malattia_A_(Exposure)": exposure, "Malattia_B_(Outcome)": outcome, **res}
                edge_rows.append(row)
            elif method == "chronos_full":
                df = run_train_variant(train_py, args, exposure, outcome, method, out_dir, extra_flags=["--use_mine"])
                row = df.iloc[0].to_dict()
                row["method"] = method
                edge_rows.append(row)
            elif method == "pred_only":
                df = run_train_variant(train_py, args, exposure, outcome, method, out_dir, extra_flags=["--alpha", "0", "--beta", "0", "--gamma", "0"])
                row = df.iloc[0].to_dict()
                row["method"] = method
                edge_rows.append(row)
            else:
                raise ValueError(f"Unknown method: {method}")

    if not edge_rows:
        raise SystemExit("No valid edge/method results were produced.")

    edge_df = pd.DataFrame(edge_rows)

    # Harmonize FDR columns and compute per-method q-values when needed.
    if "q_value" not in edge_df.columns:
        edge_df["q_value"] = np.nan
    if "global_q_value" not in edge_df.columns:
        edge_df["global_q_value"] = np.nan
    if "global_fdr_significant" not in edge_df.columns:
        edge_df["global_fdr_significant"] = False

    for method, idx in edge_df.groupby("method").groups.items():
        idx = list(idx)
        q = _bh_fdr(edge_df.loc[idx, "p_value"].astype(float).values)
        if method in {"naive_rd", "aipw_logistic"}:
            edge_df.loc[idx, "global_q_value"] = q
            edge_df.loc[idx, "global_fdr_significant"] = q < 0.05
        else:
            # keep train.py q-values if present, otherwise fill.
            missing = edge_df.loc[idx, "global_q_value"].isna()
            if missing.all():
                edge_df.loc[idx, "global_q_value"] = q
                edge_df.loc[idx, "global_fdr_significant"] = q < 0.05

    edge_df["edge"] = edge_df["Malattia_A_(Exposure)"].astype(str) + "->" + edge_df["Malattia_B_(Outcome)"].astype(str)
    cols_front = [
        "method", "edge", "Malattia_A_(Exposure)", "Malattia_B_(Outcome)",
        "Causal_Effect_ATE", "SE", "z_value", "p_value", "global_q_value", "global_fdr_significant",
        "CI_Lower_95", "CI_Upper_95", "Placebo_ATE_IPW", "N", "N_treated", "N_control",
        "ESS_ipw", "trim_frac", "e_min", "e_median", "e_max", "n_folds", "cross_fitted"
    ]
    edge_df = edge_df[[c for c in cols_front if c in edge_df.columns]]
    edge_df = edge_df.sort_values(["method", "edge"]).reset_index(drop=True)

    summary_df = summarize(edge_df)

    edge_csv = out_dir / "baseline_edge_results.csv"
    summ_csv = out_dir / "baseline_summary.csv"
    edge_df.to_csv(edge_csv, index=False)
    summary_df.to_csv(summ_csv, index=False)

    print("\nSaved:")
    print(" -", edge_csv)
    print(" -", summ_csv)
    print("\nSummary preview:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
