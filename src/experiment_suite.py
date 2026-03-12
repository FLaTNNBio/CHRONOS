import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ACTUAL_CATEGORIES = [
    "INFE", "NEOP", "ENDO", "BLD", "MENT", "NERV", "CIRC", "RESP",
    "DIGE", "GEN", "SKIN", "MUSC", "CONG", "PERI", "INJ", "SUPP",
]

TOP_INTERPRETABLE_EDGES: List[Tuple[str, str]] = [
    ("BLD", "ENDO"),
    ("NEOP", "GEN"),
    ("MENT", "MUSC"),
    ("ENDO", "DIGE"),
    ("NEOP", "DIGE"),
    ("INJ", "DIGE"),
    ("MENT", "NERV"),
    ("NERV", "GEN"),
]

CORE_SENSITIVITY_EDGES: List[Tuple[str, str]] = [
    ("BLD", "ENDO"),
    ("MENT", "MUSC"),
    ("ENDO", "DIGE"),
    ("NERV", "GEN"),
]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    alpha: float = 0.5
    use_mine: bool = True
    beta: float = 0.1
    gamma: float = 0.01
    weighted_sampling: bool = True


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def launch(cmd: List[str], cwd: str) -> int:
    print("\n$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    return int(proc.returncode)


def iter_all_dx_edges(include_self_edges: bool = False) -> Iterable[Tuple[str, str]]:
    for a in ACTUAL_CATEGORIES:
        for b in ACTUAL_CATEGORIES:
            if not include_self_edges and a == b:
                continue
            yield a, b


def parse_edge_list(raw: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "->" not in item:
            raise ValueError(f"Edge non valida: {item}. Usa formato A->B")
        a, b = item.split("->", 1)
        out.append((a.strip().upper(), b.strip().upper()))
    if not out:
        raise ValueError("Nessun edge valido specificato.")
    return out


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_train_command(
    python_exe: str,
    data_path: str,
    out_path: str,
    edge: Tuple[str, str],
    spec: ExperimentSpec,
    args,
    *,
    buffer_days: int,
    followup_days: int,
    control_ratio: int,
) -> List[str]:
    a, b = edge
    cmd = [
        python_exe,
        "train.py",
        "--data", data_path,
        "--treatment_codes", a,
        "--outcome_codes", b,
        "--out_path", out_path,
        "--epochs", str(args.epochs),
        "--n_folds", str(args.n_folds),
        "--batch_size", str(args.batch_size),
        "--min_treated", str(args.min_treated),
        "--latent_dim", str(args.latent_dim),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--max_seq_len", str(args.max_seq_len),
        "--num_workers", str(args.num_workers),
        "--baseline_window", str(args.baseline_window),
        "--followup_window", str(followup_days),
        "--washout_treatment", str(args.washout_treatment),
        "--outcome_washout", str(args.outcome_washout),
        "--buffer_days", str(buffer_days),
        "--active_obs_days", str(args.active_obs_days),
        "--control_ratio", str(control_ratio),
        "--alpha", str(spec.alpha),
        "--beta", str(spec.beta),
        "--gamma", str(spec.gamma),
        "--lambda_prop", str(args.lambda_prop),
        "--perc", str(args.perc),
        "--margin", str(args.margin),
        "--max_pairs", str(args.max_pairs),
        "--mine_interval", str(args.mine_interval),
        "--critic_steps", str(args.critic_steps),
        "--ps_eps", str(args.ps_eps),
        "--seed", str(args.seed),
        "--add_qvalues",
    ]
    if spec.use_mine:
        cmd.append("--use_mine")
    if spec.weighted_sampling:
        cmd.append("--weighted_sampling")
    return cmd


def collect_csvs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def merge_master(csv_paths: Sequence[str], out_csv: str, fdr_alpha: float):
    if not csv_paths:
        print(f"⚠️ Nessun CSV da aggregare per {out_csv}")
        return
    dfs = []
    for fp in csv_paths:
        try:
            dfs.append(pd.read_csv(fp))
        except Exception as e:
            print(f"[WARN] impossibile leggere {fp}: {e}")
    if not dfs:
        print(f"⚠️ Nessun CSV leggibile per {out_csv}")
        return

    df = pd.concat(dfs, ignore_index=True)
    if "p_value" in df.columns:
        df["global_q_value"] = bh_qvalues(df["p_value"].values)
        df["global_fdr_significant"] = np.where(df["global_q_value"] <= fdr_alpha, "Si", "No")
    sort_cols = [c for c in ["global_fdr_significant", "Causal_Effect_ATE", "p_value"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False, False, True][: len(sort_cols)])
    df.to_csv(out_csv, index=False)
    print(f"✅ Aggregato salvato in: {out_csv}")


def run_block(
    block_name: str,
    edges: Sequence[Tuple[str, str]],
    specs: Sequence[ExperimentSpec],
    args,
    script_dir: str,
    data_path: str,
    base_out_dir: str,
    *,
    buffer_days: int,
    followup_days: int,
    control_ratio: int,
):
    block_dir = os.path.join(base_out_dir, block_name)
    ensure_dir(block_dir)
    python_exe = sys.executable
    all_block_csvs: List[str] = []

    for spec in specs:
        spec_dir = os.path.join(block_dir, spec.name)
        ensure_dir(spec_dir)
        print(f"\n====================")
        print(f"BLOCK: {block_name} | EXPERIMENT: {spec.name}")
        print(spec.description)
        print(f"Edges: {len(edges)} | buffer={buffer_days} | followup={followup_days} | ratio=1:{control_ratio}")
        print("====================")

        spec_csvs: List[str] = []
        for a, b in edges:
            out_csv = os.path.join(
                spec_dir,
                f"{spec.name}__DX_{a}_TO_{b}__d{buffer_days}_h{followup_days}_r{control_ratio}.csv",
            )
            cmd = build_train_command(
                python_exe,
                data_path,
                out_csv,
                (a, b),
                spec,
                args,
                buffer_days=buffer_days,
                followup_days=followup_days,
                control_ratio=control_ratio,
            )
            ret = launch(cmd, script_dir)
            if ret == 0 and os.path.exists(out_csv):
                spec_csvs.append(out_csv)
                all_block_csvs.append(out_csv)
            else:
                print(f"⚠️ Trial fallita o senza output: {spec.name} | {a}->{b}")

        master_csv = os.path.join(spec_dir, f"MASTER__{spec.name}.csv")
        merge_master(spec_csvs, master_csv, args.fdr_alpha)

    block_master_csv = os.path.join(block_dir, f"MASTER__{block_name}.csv")
    merge_master(all_block_csvs, block_master_csv, args.fdr_alpha)


def write_summary(base_out_dir: str):
    master_paths = sorted(glob.glob(os.path.join(base_out_dir, "**", "MASTER__*.csv"), recursive=True))
    rows = []
    for fp in master_paths:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        row = {
            "file": fp,
            "n_rows": int(len(df)),
            "n_fdr_yes": int((df.get("global_fdr_significant") == "Si").sum()) if "global_fdr_significant" in df.columns else np.nan,
            "mean_ate": float(df["Causal_Effect_ATE"].mean()) if "Causal_Effect_ATE" in df.columns and len(df) else np.nan,
            "median_abs_placebo": float(df["Placebo_ATE_IPW"].abs().median()) if "Placebo_ATE_IPW" in df.columns and len(df) else np.nan,
            "median_ess": float(df["ESS_ipw"].median()) if "ESS_ipw" in df.columns and len(df) else np.nan,
        }
        rows.append(row)
    if rows:
        pd.DataFrame(rows).sort_values("file").to_csv(os.path.join(base_out_dir, "EXPERIMENT_INDEX.csv"), index=False)
        print(f"✅ Indice esperimenti salvato in: {os.path.join(base_out_dir, 'EXPERIMENT_INDEX.csv')}")


def main():
    ap = argparse.ArgumentParser(description="Suite aggiuntiva di esperimenti CHRONOS per ablation e sensitivity.")
    ap.add_argument("--data", type=str, default="dataset_chronos_dx_only.csv")
    ap.add_argument("--out_dir", type=str, default="results_experiments")
    ap.add_argument("--mode", type=str, default="paper", choices=["paper", "quick", "full", "custom"])
    ap.add_argument("--custom_edges", type=str, default="")
    ap.add_argument("--fdr_alpha", type=float, default=0.05)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--min_treated", type=int, default=50)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_seq_len", type=int, default=60)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--baseline_window", type=int, default=180)
    ap.add_argument("--followup_window", type=int, default=365)
    ap.add_argument("--washout_treatment", type=int, default=180)
    ap.add_argument("--outcome_washout", type=int, default=30)
    ap.add_argument("--buffer_days", type=int, default=14)
    ap.add_argument("--active_obs_days", type=int, default=60)
    ap.add_argument("--control_ratio", type=int, default=3)

    ap.add_argument("--lambda_prop", type=float, default=1.0)
    ap.add_argument("--perc", type=float, default=30.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--max_pairs", type=int, default=2048)
    ap.add_argument("--mine_interval", type=int, default=5)
    ap.add_argument("--critic_steps", type=int, default=1)
    ap.add_argument("--ps_eps", type=float, default=0.01)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(args.data)
    base_out_dir = os.path.abspath(args.out_dir)
    ensure_dir(base_out_dir)

    full_spec = ExperimentSpec(
        name="chronos_full",
        description="Configurazione completa: contrastive ITE-guided + MINE + weighted sampling.",
        alpha=0.5,
        use_mine=True,
        beta=0.1,
        gamma=0.01,
        weighted_sampling=True,
    )
    no_cont = ExperimentSpec(
        name="no_contrastive",
        description="Ablation: rimuove il contrastive term impostando alpha=0.",
        alpha=0.0,
        use_mine=True,
        beta=0.1,
        gamma=0.01,
        weighted_sampling=True,
    )
    no_mi = ExperimentSpec(
        name="no_mi",
        description="Ablation: rimuove il termine MI disattivando MINE e ponendo beta=0.",
        alpha=0.5,
        use_mine=False,
        beta=0.0,
        gamma=0.0,
        weighted_sampling=True,
    )
    pred_only = ExperimentSpec(
        name="pred_only",
        description="Baseline interna: solo loss predittiva + propensity, senza contrastive e senza MINE.",
        alpha=0.0,
        use_mine=False,
        beta=0.0,
        gamma=0.0,
        weighted_sampling=True,
    )

    if args.mode == "quick":
        edges_ablation = TOP_INTERPRETABLE_EDGES
        run_block(
            "ablation",
            edges_ablation,
            [full_spec, no_cont, no_mi, pred_only],
            args,
            script_dir,
            data_path,
            base_out_dir,
            buffer_days=args.buffer_days,
            followup_days=args.followup_window,
            control_ratio=args.control_ratio,
        )
    elif args.mode == "paper":
        run_block(
            "ablation",
            TOP_INTERPRETABLE_EDGES,
            [full_spec, no_cont, no_mi, pred_only],
            args,
            script_dir,
            data_path,
            base_out_dir,
            buffer_days=args.buffer_days,
            followup_days=args.followup_window,
            control_ratio=args.control_ratio,
        )
        for buf in [14, 30]:
            run_block(
                f"sensitivity_buffer_{buf}",
                CORE_SENSITIVITY_EDGES,
                [full_spec],
                args,
                script_dir,
                data_path,
                base_out_dir,
                buffer_days=buf,
                followup_days=args.followup_window,
                control_ratio=args.control_ratio,
            )
        for horizon in [365, 730]:
            run_block(
                f"sensitivity_followup_{horizon}",
                CORE_SENSITIVITY_EDGES,
                [full_spec],
                args,
                script_dir,
                data_path,
                base_out_dir,
                buffer_days=args.buffer_days,
                followup_days=horizon,
                control_ratio=args.control_ratio,
            )
        for ratio in [1, 3]:
            run_block(
                f"sensitivity_ratio_{ratio}",
                CORE_SENSITIVITY_EDGES,
                [full_spec],
                args,
                script_dir,
                data_path,
                base_out_dir,
                buffer_days=args.buffer_days,
                followup_days=args.followup_window,
                control_ratio=ratio,
            )
    elif args.mode == "full":
        edges = list(iter_all_dx_edges(include_self_edges=False))
        run_block(
            "all_dx_edges_full",
            edges,
            [full_spec],
            args,
            script_dir,
            data_path,
            base_out_dir,
            buffer_days=args.buffer_days,
            followup_days=args.followup_window,
            control_ratio=args.control_ratio,
        )
    elif args.mode == "custom":
        if not args.custom_edges:
            raise ValueError("In modalità custom devi passare --custom_edges 'A->B,C->D'")
        edges = parse_edge_list(args.custom_edges)
        run_block(
            "custom",
            edges,
            [full_spec, no_cont, no_mi, pred_only],
            args,
            script_dir,
            data_path,
            base_out_dir,
            buffer_days=args.buffer_days,
            followup_days=args.followup_window,
            control_ratio=args.control_ratio,
        )

    write_summary(base_out_dir)
    print("\n✅ Suite completata.")


if __name__ == "__main__":
    main()
