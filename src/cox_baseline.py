import argparse
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

from dataset import (
    _has_active_obs,
    _has_any_code_in_window,
    _load_and_cache_events,
    _parse_code_list,
)


ACTUAL_CATEGORIES = [
    "INFE", "NEOP", "ENDO", "BLD", "MENT", "NERV", "CIRC", "RESP",
    "DIGE", "GEN", "SKIN", "MUSC", "CONG", "PERI", "INJ", "SUPP",
]


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


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _prepare_patient_events(df: pd.DataFrame, history_col: str, outcome_col: str) -> Dict[str, Dict]:
    cols_needed = [
        "CODICE_FISCALE_ASSISTITO", "DATA_PRESCRIZIONE", history_col, outcome_col,
        "ANNO_NASCITA", "FRAILTY_SUM", "SESSO_NUM", "DATA_REVOCA"
    ]
    cols_needed = list(dict.fromkeys(cols_needed))
    work = df[cols_needed].copy()
    work = work.dropna(subset=[history_col])
    work = work.sort_values(["CODICE_FISCALE_ASSISTITO", "DATA_PRESCRIZIONE"])

    grouped = work.groupby("CODICE_FISCALE_ASSISTITO", sort=False)
    patient_events: Dict[str, Dict] = {}
    for pid, g in grouped:
        hist_vals = g[history_col]
        out_vals = g[outcome_col]
        if isinstance(hist_vals, pd.DataFrame):
            hist_vals = hist_vals.iloc[:, 0]
        if isinstance(out_vals, pd.DataFrame):
            out_vals = out_vals.iloc[:, 0]

        patient_events[pid] = {
            "dates": g["DATA_PRESCRIZIONE"].values.astype("datetime64[ns]"),
            "hist_codes": hist_vals.astype(str).to_numpy().reshape(-1),
            "out_codes": out_vals.astype(str).to_numpy().reshape(-1),
            "frailty": g["FRAILTY_SUM"].values.astype(float),
            "birth_year": float(g["ANNO_NASCITA"].iloc[0]) if pd.notna(g["ANNO_NASCITA"].iloc[0]) else np.nan,
            "sex_num": float(g["SESSO_NUM"].iloc[0]) if pd.notna(g["SESSO_NUM"].iloc[0]) else 0.5,
            "data_revoca": pd.to_datetime(g["DATA_REVOCA"].iloc[0], errors="coerce"),
        }
    return patient_events


def _sample_emulated_cohort(
    patient_events: Dict[str, Dict],
    treatment_codes: Sequence[str],
    outcome_codes: Sequence[str],
    t0_trigger_codes: Optional[Sequence[str]],
    *,
    baseline_window_days: int,
    followup_window_days: int,
    washout_treatment_days: int,
    outcome_washout_days: int,
    buffer_days: int,
    active_obs_days: int,
    control_ratio: int,
    seed: int,
) -> List[Dict]:
    rng = np.random.default_rng(seed)

    first_visit = {pid: data["dates"].min() for pid, data in patient_events.items()}
    all_pids = np.array(list(first_visit.keys()))
    all_fvs = np.array([first_visit[p] for p in all_pids])
    order = np.argsort(all_fvs)
    pids_sorted = all_pids[order]
    fv_sorted_dates = all_fvs[order]

    treat_code_set = set(treatment_codes)
    out_code_set = set(outcome_codes)
    t0_trigger_set = set(t0_trigger_codes or treatment_codes)

    treated_cohort: List[Dict] = []
    for pid, data in patient_events.items():
        dates = data["dates"]
        hist_codes = data["hist_codes"]
        out_codes_arr = data["out_codes"]
        trigger_mask = np.isin(hist_codes, list(t0_trigger_set))
        t0_dates = dates[trigger_mask]
        if len(t0_dates) == 0:
            continue

        eligibility_start = first_visit[pid] + np.timedelta64(int(baseline_window_days), "D")
        valid_t0 = None
        for t0 in t0_dates:
            if t0 < eligibility_start:
                continue
            if not _has_active_obs(dates, t0, active_obs_days):
                continue

            wash_start = t0 - np.timedelta64(int(washout_treatment_days), "D")
            if _has_any_code_in_window(dates, hist_codes, wash_start, t0, treat_code_set):
                continue

            data_revoca = data["data_revoca"]
            if not pd.isna(data_revoca):
                fu_end = t0 + np.timedelta64(int(buffer_days + followup_window_days), "D")
                if fu_end > data_revoca.to_datetime64():
                    continue

            if outcome_washout_days is None or outcome_washout_days <= 0:
                if _has_any_code_in_window(dates, out_codes_arr, dates.min(), t0, out_code_set):
                    continue
            else:
                wb_start = t0 - np.timedelta64(int(outcome_washout_days), "D")
                if _has_any_code_in_window(dates, out_codes_arr, wb_start, t0, out_code_set):
                    continue

            valid_t0 = t0
            break

        if valid_t0 is not None:
            treated_cohort.append({"pid": pid, "T0": valid_t0, "T": 1})

    control_cohort: List[Dict] = []
    for tr in treated_cohort:
        t0 = tr["T0"]
        cutoff_np = t0 - np.timedelta64(int(baseline_window_days), "D")
        idx = np.searchsorted(fv_sorted_dates, cutoff_np, side="right")
        pool = pids_sorted[:idx]
        if len(pool) == 0:
            continue

        sampled_controls = []
        candidate_pool = rng.choice(pool, size=min(2000, len(pool)), replace=False)
        for c_pid in candidate_pool:
            if c_pid == tr["pid"]:
                continue
            c_data = patient_events[c_pid]
            c_dates = c_data["dates"]
            c_hist = c_data["hist_codes"]
            c_out = c_data["out_codes"]

            if not _has_active_obs(c_dates, t0, active_obs_days):
                continue

            wash_start = t0 - np.timedelta64(int(washout_treatment_days), "D")
            if _has_any_code_in_window(c_dates, c_hist, wash_start, t0 + np.timedelta64(1, 'ns'), treat_code_set):
                continue

            c_revoca = c_data["data_revoca"]
            if not pd.isna(c_revoca):
                fu_end = t0 + np.timedelta64(int(buffer_days + followup_window_days), "D")
                if fu_end > c_revoca.to_datetime64():
                    continue

            if outcome_washout_days is None or outcome_washout_days <= 0:
                if _has_any_code_in_window(c_dates, c_out, c_dates.min(), t0, out_code_set):
                    continue
            else:
                wb_start = t0 - np.timedelta64(int(outcome_washout_days), "D")
                if _has_any_code_in_window(c_dates, c_out, wb_start, t0, out_code_set):
                    continue

            sampled_controls.append({"pid": c_pid, "T0": t0, "T": 0})
            if len(sampled_controls) >= int(control_ratio):
                break

        control_cohort.extend(sampled_controls)

    return treated_cohort + control_cohort


def build_survival_dataframe(
    csv_path: str,
    treatment_codes: Sequence[str],
    outcome_codes: Sequence[str],
    *,
    t0_trigger_codes: Optional[Sequence[str]] = None,
    baseline_window_days: int = 180,
    followup_window_days: int = 365,
    washout_treatment_days: int = 180,
    outcome_washout_days: int = 30,
    buffer_days: int = 14,
    active_obs_days: int = 60,
    control_ratio: int = 3,
    min_treated: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    df = _load_and_cache_events(csv_path)

    treatment_codes = _parse_code_list(treatment_codes)
    outcome_codes = _parse_code_list(outcome_codes)
    t0_trigger_codes = _parse_code_list(t0_trigger_codes) or treatment_codes

    if not treatment_codes:
        raise ValueError("Pass at least one treatment code.")
    if not outcome_codes:
        raise ValueError("Pass at least one outcome code.")
    if len(outcome_codes) != 1:
        raise ValueError("This Cox baseline script expects exactly one outcome code per run.")
    if len(treatment_codes) != 1:
        raise ValueError("This Cox baseline script expects exactly one treatment code per run.")

    treatment_is_drug = any(c.startswith("DRUG:") for c in treatment_codes)
    history_col = "DRUG_CODE" if treatment_is_drug else "ICD9_CM"
    outcome_col = "ICD9_CM"

    patient_events = _prepare_patient_events(df, history_col, outcome_col)
    cohort = _sample_emulated_cohort(
        patient_events,
        treatment_codes=treatment_codes,
        outcome_codes=outcome_codes,
        t0_trigger_codes=t0_trigger_codes,
        baseline_window_days=baseline_window_days,
        followup_window_days=followup_window_days,
        washout_treatment_days=washout_treatment_days,
        outcome_washout_days=outcome_washout_days,
        buffer_days=buffer_days,
        active_obs_days=active_obs_days,
        control_ratio=control_ratio,
        seed=seed,
    )

    n_treated = sum(int(x["T"]) for x in cohort)
    if n_treated < min_treated:
        raise ValueError(f"Only {n_treated} treated observations after emulation (minimum required: {min_treated}).")

    out_code = outcome_codes[0]
    rows: List[Dict] = []
    for row in cohort:
        pid = row["pid"]
        t0 = row["T0"].astype("datetime64[ns]")
        is_treated = int(row["T"])
        data = patient_events[pid]
        dates = data["dates"]
        out_codes_arr = data["out_codes"]

        t0_idx = np.searchsorted(dates, t0, side="right") - 1
        frailty_val = float(data["frailty"][t0_idx]) if t0_idx >= 0 else 0.0
        if not np.isfinite(frailty_val):
            frailty_val = 0.0

        birth_year = _safe_float(data["birth_year"])
        t0_year = pd.Timestamp(t0).year
        age_at_t0 = float(t0_year - birth_year) if np.isfinite(birth_year) else np.nan
        sex_num = _safe_float(data["sex_num"], default=0.5)

        baseline_start = t0 - np.timedelta64(int(baseline_window_days), "D")
        b_mask = (dates >= baseline_start) & (dates < t0)
        b_dates = dates[b_mask]
        b_out_codes = out_codes_arr[b_mask]
        prior_outcome_count = float(np.sum(b_out_codes == out_code))
        util_last30 = float(np.sum((dates >= (t0 - np.timedelta64(30, 'D'))) & (dates < t0)))

        risk_start = t0 + np.timedelta64(int(buffer_days), "D")
        risk_end = risk_start + np.timedelta64(int(followup_window_days), "D")
        f_mask = (dates > risk_start) & (dates <= risk_end)
        f_dates = dates[f_mask]
        f_codes = out_codes_arr[f_mask]
        event_dates = f_dates[f_codes == out_code]

        if len(event_dates) > 0:
            event_date = event_dates.min()
            duration_days = float((event_date - risk_start) / np.timedelta64(1, "D"))
            event = 1
        else:
            duration_days = float((risk_end - risk_start) / np.timedelta64(1, "D"))
            event = 0

        duration_days = max(duration_days, 1e-6)

        rows.append({
            "pid": pid,
            "treatment": treatment_codes[0],
            "outcome": out_code,
            "T": is_treated,
            "duration": duration_days,
            "event": event,
            "age_at_t0": age_at_t0,
            "sex_num": sex_num,
            "frailty": frailty_val,
            "prior_outcome_count": prior_outcome_count,
            "util_last30": util_last30,
            "t0": pd.Timestamp(t0),
            "risk_start": pd.Timestamp(risk_start),
            "risk_end": pd.Timestamp(risk_end),
        })

    sdf = pd.DataFrame(rows)
    sdf["age_at_t0"] = sdf["age_at_t0"].fillna(sdf["age_at_t0"].median())
    return sdf


def _usable_exog_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    keep: List[str] = []
    X = df[list(cols)].copy()
    for c in cols:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    # drop exactly duplicated columns
    final_keep: List[str] = []
    for c in keep:
        dup = False
        for k in final_keep:
            if np.allclose(X[c].to_numpy(dtype=float), X[k].to_numpy(dtype=float), equal_nan=True):
                dup = True
                break
        if not dup:
            final_keep.append(c)
    return final_keep


def _check_survival_identifiability(df: pd.DataFrame) -> Optional[str]:
    n = int(len(df))
    if n < 20:
        return f"too few rows ({n})"
    if int(df['event'].sum()) == 0:
        return 'no outcome events in follow-up'
    if df['T'].nunique(dropna=True) < 2:
        return 'treatment has no variation'
    n_t = int((df['T'] == 1).sum())
    n_c = int((df['T'] == 0).sum())
    if n_t == 0 or n_c == 0:
        return 'missing treated or control arm'
    e_t = int(df.loc[df['T'] == 1, 'event'].sum())
    e_c = int(df.loc[df['T'] == 0, 'event'].sum())
    if e_t == 0 and e_c == 0:
        return 'no events in either arm'
    return None


def fit_single_cox(df: pd.DataFrame, ties: str = "breslow", robust: bool = True):
    ident_err = _check_survival_identifiability(df)
    if ident_err is not None:
        raise ValueError(ident_err)

    candidate_sets = [
        ["T", "age_at_t0", "sex_num", "frailty", "prior_outcome_count", "util_last30"],
        ["T", "age_at_t0", "sex_num", "frailty", "util_last30"],
        ["T", "age_at_t0", "sex_num"],
        ["T"],
    ]

    last_err = None
    for cand in candidate_sets:
        exog_cols = _usable_exog_columns(df, cand)
        if "T" not in exog_cols:
            exog_cols = ["T"] + [c for c in exog_cols if c != "T"]
        exog = df[exog_cols].astype(float)
        if exog.shape[1] == 0:
            continue
        if np.linalg.matrix_rank(exog.to_numpy()) < exog.shape[1]:
            keep = ["T"]
            for c in exog_cols:
                if c == "T":
                    continue
                test_cols = keep + [c]
                Xtest = df[test_cols].astype(float).to_numpy()
                if np.linalg.matrix_rank(Xtest) == len(test_cols):
                    keep.append(c)
            exog_cols = keep
            exog = df[exog_cols].astype(float)

        model = PHReg(
            endog=df["duration"].astype(float).to_numpy(),
            exog=exog,
            status=df["event"].astype(int).to_numpy(),
            ties=ties,
        )

        for robust_flag in ([robust, False] if robust else [False]):
            try:
                res = model.fit(groups=df["pid"].astype(str).to_numpy() if robust_flag else None, disp=False)
                params = pd.Series(np.asarray(res.params).reshape(-1), index=exog_cols)
                bse = pd.Series(np.asarray(res.bse).reshape(-1), index=exog_cols)
                pvalues = pd.Series(np.asarray(res.pvalues).reshape(-1), index=exog_cols)

                treatment_beta = float(params["T"])
                treatment_se = float(bse["T"])
                hr = math.exp(treatment_beta)
                ci_low = math.exp(treatment_beta - 1.96 * treatment_se)
                ci_high = math.exp(treatment_beta + 1.96 * treatment_se)

                finite_ok = np.isfinite([treatment_beta, treatment_se, hr, ci_low, ci_high, float(pvalues["T"])]).all()
                if not finite_ok:
                    raise ValueError("non-finite Cox estimates")

                out = {
                    "coef_log_hazard": treatment_beta,
                    "se": treatment_se,
                    "hazard_ratio": hr,
                    "ci_lower_95": ci_low,
                    "ci_upper_95": ci_high,
                    "p_value": float(pvalues["T"]),
                    "n": int(len(df)),
                    "n_treated": int(df["T"].sum()),
                    "n_control": int((1 - df["T"]).sum()),
                    "events": int(df["event"].sum()),
                    "treated_events": int(df.loc[df["T"] == 1, "event"].sum()),
                    "control_events": int(df.loc[df["T"] == 0, "event"].sum()),
                    "treated_event_rate": float(df.loc[df["T"] == 1, "event"].mean()) if (df["T"] == 1).any() else np.nan,
                    "control_event_rate": float(df.loc[df["T"] == 0, "event"].mean()) if (df["T"] == 0).any() else np.nan,
                    "median_followup_days": float(df["duration"].median()),
                    "ties": ties,
                    "robust_cluster_pid": bool(robust_flag),
                    "age_mean": float(df["age_at_t0"].mean()),
                    "frailty_mean": float(df["frailty"].mean()),
                    "prior_outcome_count_mean": float(df["prior_outcome_count"].mean()),
                    "util_last30_mean": float(df["util_last30"].mean()),
                    "cox_covariates": ";".join(exog_cols),
                }
                return out, res
            except Exception as e:
                last_err = e
                continue

    raise ValueError(str(last_err) if last_err is not None else "Cox fit failed")


def parse_edge_strings(edges: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for edge in edges:
        if "->" not in edge:
            raise ValueError(f"Invalid edge format: {edge}. Use A->B, e.g. BLD->ENDO")
        a, b = edge.split("->", 1)
        parsed.append((a.strip().upper(), b.strip().upper()))
    return parsed


def load_edges(args) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    if args.edge:
        edges.extend(parse_edge_strings(args.edge))
    if args.edges_csv:
        tab = pd.read_csv(args.edges_csv)
        if {"A", "B"}.issubset(tab.columns):
            edges.extend([(str(a).strip().upper(), str(b).strip().upper()) for a, b in zip(tab["A"], tab["B"])])
        elif {"Malattia_A_(Exposure)", "Malattia_B_(Outcome)"}.issubset(tab.columns):
            edges.extend([
                (str(a).strip().upper(), str(b).strip().upper())
                for a, b in zip(tab["Malattia_A_(Exposure)"], tab["Malattia_B_(Outcome)"])
            ])
        else:
            raise ValueError("edges_csv must contain either columns [A,B] or [Malattia_A_(Exposure), Malattia_B_(Outcome)].")
    if args.top_from_master:
        master = pd.read_csv(args.top_from_master)
        if "Causal_Effect_ATE" in master.columns:
            master = master.sort_values("Causal_Effect_ATE", ascending=False)
        if args.only_fdr_yes and "Significativo_FDR" in master.columns:
            master = master[master["Significativo_FDR"].astype(str).str.lower().isin(["si", "yes", "true"])]
        master = master.head(args.top_k)
        edges.extend([
            (str(a).strip().upper(), str(b).strip().upper())
            for a, b in zip(master["Malattia_A_(Exposure)"], master["Malattia_B_(Outcome)"])
        ])
    if not edges:
        raise ValueError("No edges specified. Use --edge, --edges_csv, or --top_from_master.")
    return list(dict.fromkeys(edges))


def main():
    p = argparse.ArgumentParser(description="Cox baseline epidemiological analysis for selected CHRONOS trajectories.")
    p.add_argument("--data", type=str, default="dataset_chronos.csv")
    p.add_argument("--out_dir", type=str, default="cox_baseline_results")
    p.add_argument("--seed", type=int, default=42)

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--edge", type=str, nargs="+", help="One or more edges in the form A->B, e.g. BLD->ENDO")
    group.add_argument("--edges_csv", type=str, help="CSV with columns A,B or Malattia_A_(Exposure),Malattia_B_(Outcome)")
    group.add_argument("--top_from_master", type=str, help="MASTER_ATE_Graph_FDR.csv to select top edges from CHRONOS results")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--only_fdr_yes", action="store_true")

    p.add_argument("--baseline_window", type=int, default=180)
    p.add_argument("--followup_window", type=int, default=365)
    p.add_argument("--washout_treatment", type=int, default=180)
    p.add_argument("--outcome_washout", type=int, default=30)
    p.add_argument("--buffer_days", type=int, default=14)
    p.add_argument("--active_obs_days", type=int, default=60)
    p.add_argument("--control_ratio", type=int, default=3)
    p.add_argument("--min_treated", type=int, default=50)

    p.add_argument("--ties", type=str, default="breslow", choices=["breslow", "efron"])
    p.add_argument("--no_robust", action="store_true", help="Disable cluster-robust SE by patient id")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    edges = load_edges(args)

    summaries = []
    for a, b in edges:
        print(f"\n>>> Cox baseline for {a} -> {b}")
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
            edge_df_path = os.path.join(args.out_dir, f"cox_cohort_{a}_TO_{b}.csv")
            sdf.to_csv(edge_df_path, index=False)

            summary, _ = fit_single_cox(sdf, ties=args.ties, robust=not args.no_robust)
            summary.update({
                "Exposure": a,
                "Outcome": b,
                "cohort_csv": edge_df_path,
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
                "Exposure": a,
                "Outcome": b,
                "error": str(e),
            })

    out = pd.DataFrame(summaries)
    if "p_value" in out.columns:
        mask = out["p_value"].notna()
        out.loc[mask, "q_value"] = bh_qvalues(out.loc[mask, "p_value"].to_numpy())
        out["Significativo_FDR"] = np.where(out.get("q_value", 1.0) <= 0.05, "Si", "No")

    cols = [
        "Exposure", "Outcome", "hazard_ratio", "ci_lower_95", "ci_upper_95", "p_value", "q_value",
        "Significativo_FDR", "n", "n_treated", "n_control", "events", "treated_events", "control_events",
        "treated_event_rate", "control_event_rate", "median_followup_days", "coef_log_hazard", "se",
        "age_mean", "frailty_mean", "prior_outcome_count_mean", "util_last30_mean", "ties",
        "robust_cluster_pid", "cohort_csv", "error"
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]
    summary_path = os.path.join(args.out_dir, "cox_baseline_summary.csv")
    out.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
