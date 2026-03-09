import argparse
import math
import os
import random
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import ChronosTargetTrialDataset, build_weighted_sampler, collate_fn
from model import CHRONOSModel, MineStatisticsNetwork, contrastive_loss_from_batch, mine_lower_bound_stable


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, indices: Sequence[int], batch_size: int, num_workers: int, weighted_sampling: bool, shuffle: bool):
    subset = Subset(dataset, list(map(int, indices)))
    sampler = build_weighted_sampler(dataset, indices) if weighted_sampling else None
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(dataset, args, device):
    return CHRONOSModel(
        cov_dim=dataset.cov_dim,
        prev_treat_dim=dataset.n_treatments,
        latent_dim=args.latent_dim,
        rnn_layers=1,
        dropout=args.dropout,
        n_treatments=dataset.n_treatments,
        n_outcomes=dataset.n_outcomes,
        static_dim=3,
    ).to(device)


def fit_fold(model, train_loader, args, device):
    mine_bias = MineStatisticsNetwork(x_dim=args.latent_dim, y_dim=1).to(device)
    mine_out = MineStatisticsNetwork(x_dim=args.latent_dim, y_dim=train_loader.dataset.dataset.n_outcomes).to(device)

    opt_model = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_mine_b = torch.optim.Adam(mine_bias.parameters(), lr=args.lr)
    opt_mine_o = torch.optim.Adam(mine_out.parameters(), lr=args.lr)
    bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            global_step += 1
            cov = batch["current_covariates"].to(device)
            prev_tr = batch["prev_treatments"].to(device)
            cur_tr = batch["current_treatments"].to(device)
            y = batch["outputs"].to(device)
            seq_lens = batch["seq_len"].to(device)
            treated = batch["treated"].to(device)

            logits, z, _, e_logit = model(cov, prev_tr, cur_tr)
            batch_idx = torch.arange(logits.size(0), device=device)
            target_idx = torch.clamp(seq_lens - 2, min=0)

            logits_target = logits[batch_idx, target_idx]
            y_target = y[batch_idx, target_idx]
            z_target = z[batch_idx, target_idx]
            e_logit_target = e_logit[batch_idx, target_idx].squeeze(-1)
            tr_target = cur_tr[batch_idx, target_idx]

            loss_sup = bce_logits(logits_target, y_target).mean(dim=-1).mean()
            loss_prop = nn.functional.binary_cross_entropy_with_logits(e_logit_target, treated)

            with torch.no_grad():
                y0, y1_all = model.predict_outcomes(z_target)
                tau = torch.sigmoid(y1_all[:, 0, :]) - torch.sigmoid(y0)
            loss_cont = contrastive_loss_from_batch(
                z_target, tau, max_pairs=args.max_pairs, margin=args.margin, perc=args.perc
            )

            if args.use_mine and (global_step % args.mine_interval == 0):
                z_detach = z_target.detach()
                for _ in range(args.critic_steps):
                    opt_mine_b.zero_grad()
                    mi_b = mine_lower_bound_stable(mine_bias, z_detach.float(), tr_target.float())
                    (-mi_b).backward()
                    opt_mine_b.step()

                    opt_mine_o.zero_grad()
                    mi_o = mine_lower_bound_stable(mine_out, z_detach.float(), y_target.float())
                    (-mi_o).backward()
                    opt_mine_o.step()

                mi_b_main = mine_lower_bound_stable(mine_bias, z_target.float(), tr_target.float())
                mi_o_main = mine_lower_bound_stable(mine_out, z_target.float(), y_target.float())
                loss = loss_sup + args.lambda_prop * loss_prop + args.alpha * loss_cont + args.beta * mi_b_main - args.gamma * mi_o_main
            else:
                loss = loss_sup + args.lambda_prop * loss_prop + args.alpha * loss_cont

            opt_model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_model.step()
            total_loss += float(loss.item())

        print(f"Fold epoch [{epoch:03d}/{args.epochs:03d}] Loss: {total_loss / max(1, len(train_loader)):.4f}")


def infer_fold(model, test_loader, device):
    model.eval()
    mu0_all, mu1_all, e_all, T_all, Y_all, placebo_all, pids_all = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            cov = batch["current_covariates"].to(device)
            prev_tr = batch["prev_treatments"].to(device)
            cur_tr = batch["current_treatments"].to(device)
            y = batch["outputs"].to(device)
            seq_lens = batch["seq_len"].to(device)
            T = batch["treated"].to(device)
            placebo = batch["placebo_outcome"].to(device)

            _, z, _, e_logit = model(cov, prev_tr, cur_tr)
            batch_idx = torch.arange(z.size(0), device=device)
            target_idx = torch.clamp(seq_lens - 2, min=0)
            z_target = z[batch_idx, target_idx]
            y_target = y[batch_idx, target_idx]
            e_target = torch.sigmoid(e_logit[batch_idx, target_idx].squeeze(-1))
            y0, y1_all = model.predict_outcomes(z_target)
            mu0 = torch.sigmoid(y0)
            mu1 = torch.sigmoid(y1_all[:, 0, :])

            mu0_all.append(mu0.cpu().numpy())
            mu1_all.append(mu1.cpu().numpy())
            e_all.append(e_target.cpu().numpy())
            T_all.append(T.cpu().numpy())
            Y_all.append(y_target.cpu().numpy())
            placebo_all.append(placebo.cpu().numpy())
            pids_all.extend(batch["pids"])

    return {
        "pid": np.asarray(pids_all),
        "mu0": np.concatenate(mu0_all, axis=0),
        "mu1": np.concatenate(mu1_all, axis=0),
        "e": np.concatenate(e_all, axis=0),
        "T": np.concatenate(T_all, axis=0).astype(float),
        "Y": np.concatenate(Y_all, axis=0),
        "placebo": np.concatenate(placebo_all, axis=0),
    }


def make_folds(n: int, n_folds: int, seed: int) -> List[np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_folds = max(2, min(int(n_folds), n))
    return [fold.astype(int) for fold in np.array_split(idx, n_folds) if len(fold) > 0]


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tte_args = {
        "t0_trigger_codes": args.t0_codes.split(",") if args.t0_codes else None,
        "treatment_codes": args.treatment_codes.split(",") if args.treatment_codes else None,
        "outcome_codes": args.outcome_codes.split(",") if args.outcome_codes else None,
        "baseline_window_days": args.baseline_window,
        "followup_window_days": args.followup_window,
        "washout_treatment_days": args.washout_treatment,
        "outcome_washout_days": args.outcome_washout,
        "buffer_days": args.buffer_days,
        "active_obs_days": args.active_obs_days,
        "control_ratio": args.control_ratio,
        "seed": args.seed,
    }

    dataset = ChronosTargetTrialDataset(args.data, max_seq_len=args.max_seq_len, **tte_args)
    if len(dataset) == 0:
        print("\n⚠️ ABORTING TRIAL: 0 patients after target-trial emulation.")
        return
    if dataset.n_treated < args.min_treated:
        print(f"\n⚠️ ABORTING TRIAL: only {dataset.n_treated} treated patients (minimum: {args.min_treated}).")
        return
    if dataset.n_treatments != 1:
        print("\n⚠️ ABORTING TRIAL: this implementation supports one treatment code per edge-specific trial.")
        return

    n = len(dataset)
    folds = make_folds(n, args.n_folds, args.seed)
    print(f"Starting {len(folds)}-fold cross-fitting on n={n} patients...")

    fold_outputs: List[Dict] = []
    for fold_id, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        print(f"\n--- Fold {fold_id}/{len(folds)} | train={len(train_idx)} | test={len(test_idx)} ---")
        train_loader = make_loader(dataset, train_idx, args.batch_size, args.num_workers, args.weighted_sampling, shuffle=True)
        test_loader = make_loader(dataset, test_idx, args.batch_size, args.num_workers, False, shuffle=False)
        model = build_model(dataset, args, device)
        fit_fold(model, train_loader, args, device)
        fold_pred = infer_fold(model, test_loader, device)
        fold_pred["fold"] = np.full(len(test_idx), fold_id, dtype=int)
        fold_outputs.append(fold_pred)

    if not fold_outputs:
        print("\n⚠️ ABORTING TRIAL: no valid folds were produced.")
        return

    mu0 = np.concatenate([d["mu0"] for d in fold_outputs], axis=0)
    mu1 = np.concatenate([d["mu1"] for d in fold_outputs], axis=0)
    e = np.concatenate([d["e"] for d in fold_outputs], axis=0)
    T = np.concatenate([d["T"] for d in fold_outputs], axis=0).astype(float)
    Y = np.concatenate([d["Y"] for d in fold_outputs], axis=0)
    Y_pl = np.concatenate([d["placebo"] for d in fold_outputs], axis=0)
    pids = np.concatenate([d["pid"] for d in fold_outputs], axis=0)
    fold_ids = np.concatenate([d["fold"] for d in fold_outputs], axis=0)

    order = np.argsort(pids.astype(str))
    mu0, mu1, e, T, Y, Y_pl, pids, fold_ids = [arr[order] for arr in (mu0, mu1, e, T, Y, Y_pl, pids, fold_ids)]

    eps = float(args.ps_eps)
    e_tilde = np.clip(e, eps, 1.0 - eps)
    trim_frac = float(np.mean((e < eps) | (e > (1.0 - eps))))

    T_col = T.reshape(-1, 1)
    e_col = e_tilde.reshape(-1, 1)
    psi = (mu1 - mu0) + (T_col / e_col) * (Y - mu1) - ((1.0 - T_col) / (1.0 - e_col)) * (Y - mu0)
    ate = psi.mean(axis=0)
    se = psi.std(axis=0, ddof=1) / math.sqrt(max(1, psi.shape[0]))
    zval = np.divide(ate, se, out=np.zeros_like(ate), where=se > 0)
    pval = 2.0 * (1.0 - _norm_cdf(np.abs(zval)))
    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se

    e_min, e_med, e_max = float(np.min(e)), float(np.median(e)), float(np.max(e))
    w = T / e_tilde + (1.0 - T) / (1.0 - e_tilde)
    ess = float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))
    psi_pl = (T_col / e_col) * Y_pl - ((1.0 - T_col) / (1.0 - e_col)) * Y_pl
    placebo_ate = psi_pl.mean(axis=0)

    exposure = dataset.treat_codes[0]
    rows = []
    for j, outcome in enumerate(dataset.out_codes):
        rows.append({
            "Malattia_A_(Exposure)": exposure,
            "Malattia_B_(Outcome)": outcome,
            "Causal_Effect_ATE": float(ate[j]),
            "SE": float(se[j]),
            "z_value": float(zval[j]),
            "p_value": float(pval[j]),
            "CI_Lower_95": float(ci_low[j]),
            "CI_Upper_95": float(ci_high[j]),
            "Placebo_ATE_IPW": float(placebo_ate[j]),
            "N": int(len(T)),
            "N_treated": int(T.sum()),
            "N_control": int(len(T) - T.sum()),
            "trim_frac": trim_frac,
            "ESS_ipw": ess,
            "e_min": e_min,
            "e_median": e_med,
            "e_max": e_max,
            "n_folds": int(len(folds)),
            "cross_fitted": True,
        })

    df = pd.DataFrame(rows).sort_values("Causal_Effect_ATE", ascending=False)
    if args.add_qvalues:
        df["q_value"] = _bh_fdr(df["p_value"].values)

    out_path = args.out_path or os.path.join("risultati_causali", f"parziale_{exposure}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="dataset_chronos.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_path", type=str, default="")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_seq_len", type=int, default=60)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_folds", type=int, default=5)

    p.add_argument("--min_treated", type=int, default=50)
    p.add_argument("--t0_codes", type=str, default="")
    p.add_argument("--treatment_codes", type=str, default="")
    p.add_argument("--outcome_codes", type=str, default="")

    p.add_argument("--baseline_window", type=int, default=180)
    p.add_argument("--followup_window", type=int, default=365)
    p.add_argument("--washout_treatment", type=int, default=180)
    p.add_argument("--outcome_washout", type=int, default=30)
    p.add_argument("--buffer_days", type=int, default=14)
    p.add_argument("--active_obs_days", type=int, default=60)
    p.add_argument("--control_ratio", type=int, default=3)

    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--lambda_prop", type=float, default=1.0)

    p.add_argument("--perc", type=float, default=30.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--max_pairs", type=int, default=2048)

    p.add_argument("--use_mine", action="store_true")
    p.add_argument("--mine_interval", type=int, default=5)
    p.add_argument("--critic_steps", type=int, default=1)
    p.add_argument("--weighted_sampling", action="store_true")

    p.add_argument("--ps_eps", type=float, default=0.01)
    p.add_argument("--add_qvalues", action="store_true")

    args = p.parse_args()
    train(args)
