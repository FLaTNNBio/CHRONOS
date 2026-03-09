import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


DEFAULT_STATIC_COLS = ["ANNO_NASCITA", "FRAILTY_SUM", "SESSO_NUM"]


def icd9_to_macro(code: str) -> str:
    """Project-specific ICD-9/ATC to macro mapping aligned with the paper."""
    s = str(code).strip().upper()
    if not s:
        return "SUPP"

    if s.startswith("DRUG:"):
        return s

    if s.startswith(("V", "E")):
        return "SUPP"

    try:
        main = s.split(".")[0]
        n = int(float(main))
    except Exception:
        return "SUPP"

    if 1 <= n <= 139:
        return "INFE"
    if 140 <= n <= 239:
        return "NEOP"
    if 240 <= n <= 279:
        return "ENDO"
    if 280 <= n <= 289:
        return "BLD"
    if 290 <= n <= 319:
        return "MENT"
    if 320 <= n <= 389:
        return "NERV"
    if 390 <= n <= 459:
        return "CIRC"
    if 460 <= n <= 519:
        return "RESP"
    if 520 <= n <= 579:
        return "DIGE"
    if 580 <= n <= 629:
        return "GEN"
    if 680 <= n <= 709:
        return "SKIN"
    if 710 <= n <= 739:
        return "MUSC"
    if 740 <= n <= 759:
        return "CONG"
    if 760 <= n <= 779:
        return "PERI"
    if 800 <= n <= 999:
        return "INJ"
    return "SUPP"


def _parse_code_list(codes: Optional[Sequence[str]]) -> Optional[List[str]]:
    if codes is None:
        return None
    out = []
    for c in codes:
        c = str(c).strip().upper()
        if c:
            out.append(c)
    return sorted(set(out)) or None


def _encode_sex(x) -> float:
    s = str(x).strip().upper()
    if s in {"M", "MALE", "1"}:
        return 1.0
    if s in {"F", "FEMALE", "0"}:
        return 0.0
    return 0.5


def _has_active_obs(dates: np.ndarray, t0: np.datetime64, active_obs_days: int) -> bool:
    if active_obs_days is None or active_obs_days <= 0:
        return True
    start = t0 - np.timedelta64(int(active_obs_days), "D")
    left = np.searchsorted(dates, start, side="left")
    right = np.searchsorted(dates, t0, side="right")
    return left < right


def _has_any_code_in_window(
    dates: np.ndarray,
    codes: np.ndarray,
    t_start: np.datetime64,
    t_end: np.datetime64,
    code_set: set,
    *,
    include_left: bool = True,
    include_right: bool = False,
) -> bool:
    if not code_set:
        return False
    left = np.searchsorted(dates, t_start, side="left" if include_left else "right")
    right = np.searchsorted(dates, t_end, side="right" if include_right else "left")
    if right <= left:
        return False
    for c in codes[left:right]:
        if c in code_set:
            return True
    return False


def _load_and_cache_events(csv_path: str) -> pd.DataFrame:
    cache_file = csv_path.replace('.csv', '_macro_cached.pkl')
    if os.path.exists(cache_file):
        print(f"🔄 Loading cached mapped dataframe: {cache_file}")
        return pd.read_pickle(cache_file)

    print("🚀 First pass: reading CSV, mapping ICD9/ATC codes, building cache...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df["DATA_PRESCRIZIONE"] = pd.to_datetime(df.get("DATA_PRESCRIZIONE"), errors="coerce")
    df = df.dropna(subset=["CODICE_FISCALE_ASSISTITO", "DATA_PRESCRIZIONE"])

    if "ICD9_CM" not in df.columns:
        df["ICD9_CM"] = np.nan
    codes = df["ICD9_CM"].astype(str).str.strip().str.upper()
    codes = codes.where(codes.ne("NAN"), np.nan)
    mapped = {c: icd9_to_macro(c) for c in codes.dropna().unique()}
    df["ICD9_CM"] = codes.map(mapped)

    if "CODICE_PRESCRIZIONE" in df.columns:
        atc = df["CODICE_PRESCRIZIONE"].astype(str).str.strip().str.upper().str[:5]
        atc = atc.where(~atc.isin(["", "NAN", "NONE"]), np.nan)
        drug_code = "DRUG:" + atc.fillna("")
        drug_code = drug_code.where(atc.notna(), np.nan)
        df["DRUG_CODE"] = drug_code
    else:
        df["DRUG_CODE"] = np.nan

    if "FRAILTY_SUM" not in df.columns:
        df["FRAILTY_SUM"] = 0.0
    df["FRAILTY_SUM"] = pd.to_numeric(df["FRAILTY_SUM"], errors="coerce").fillna(0.0)

    if "ANNO_NASCITA" not in df.columns:
        df["ANNO_NASCITA"] = np.nan
    df["ANNO_NASCITA"] = pd.to_numeric(df["ANNO_NASCITA"], errors="coerce")

    if "SESSO" not in df.columns:
        df["SESSO"] = np.nan
    df["SESSO_NUM"] = df["SESSO"].apply(_encode_sex)

    if "DATA_REVOCA" in df.columns:
        df["DATA_REVOCA"] = pd.to_datetime(df["DATA_REVOCA"], errors="coerce")
    else:
        df["DATA_REVOCA"] = pd.NaT

    df = df.sort_values(["CODICE_FISCALE_ASSISTITO", "DATA_PRESCRIZIONE"])
    df.to_pickle(cache_file)
    print(f"💾 Saved mapped cache to {cache_file}")
    return df


class ChronosTargetTrialDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        t0_trigger_codes: Optional[Sequence[str]] = None,
        treatment_codes: Optional[Sequence[str]] = None,
        outcome_codes: Optional[Sequence[str]] = None,
        *,
        baseline_window_days: int = 180,
        followup_window_days: int = 365,
        washout_treatment_days: int = 180,
        outcome_washout_days: int = 30,
        buffer_days: int = 14,
        active_obs_days: int = 60,
        control_ratio: int = 3,
        max_seq_len: int = 60,
        seed: int = 42,
        **_ignored,
    ):
        random.seed(seed)
        np.random.seed(seed)

        print(
            f"Loading {csv_path} for target-trial emulation\n"
            f"  baseline={baseline_window_days}d, followup={followup_window_days}d, "
            f"washoutA={washout_treatment_days}d, washoutB={outcome_washout_days}d, "
            f"buffer={buffer_days}d, active_obs={active_obs_days}d, ratio=1:{control_ratio}"
        )

        df = _load_and_cache_events(csv_path)
        self.max_seq_len = int(max_seq_len)
        self.patients: List[Dict] = []

        treatment_codes = _parse_code_list(treatment_codes)
        outcome_codes = _parse_code_list(outcome_codes)
        t0_trigger_codes = _parse_code_list(t0_trigger_codes) or treatment_codes

        if not treatment_codes:
            raise ValueError("Pass at least one treatment code for edge-specific target trial emulation.")
        if not outcome_codes:
            raise ValueError("Pass at least one outcome code for edge-specific target trial emulation.")

        self.treat_codes = treatment_codes
        self.out_codes = outcome_codes
        self.treat2idx = {c: i for i, c in enumerate(self.treat_codes)}
        self.out2idx = {c: i for i, c in enumerate(self.out_codes)}
        self.n_treatments = len(self.treat_codes)
        self.n_outcomes = len(self.out_codes)

        treatment_is_drug = any(c.startswith("DRUG:") for c in self.treat_codes)
        history_col = "DRUG_CODE" if treatment_is_drug else "ICD9_CM"
        outcome_col = "ICD9_CM"
        t0_col = history_col

        cols_needed = [
            "CODICE_FISCALE_ASSISTITO", "DATA_PRESCRIZIONE", history_col, outcome_col,
            "ANNO_NASCITA", "FRAILTY_SUM", "SESSO_NUM", "DATA_REVOCA"
        ]
        # When history_col == outcome_col (e.g. DX->DX trials, both on ICD9_CM),
        # pandas would otherwise keep duplicate columns and g[history_col] would
        # return a DataFrame instead of a Series. That later breaks boolean indexing
        # with "too many indices". Keep unique columns while preserving order.
        cols_needed = list(dict.fromkeys(cols_needed))
        work = df[cols_needed].copy()
        work = work.dropna(subset=[history_col])
        work = work.sort_values(["CODICE_FISCALE_ASSISTITO", "DATA_PRESCRIZIONE"])

        grouped = work.groupby("CODICE_FISCALE_ASSISTITO", sort=False)
        patient_events: Dict[str, Dict] = {}
        for pid, g in grouped:
            hist_vals = g[history_col]
            out_vals = g[outcome_col]
            # Extra guard against accidental duplicated column names / 2D values.
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

        birth_years = np.array([v["birth_year"] for v in patient_events.values()], dtype=float)
        valid_birth = birth_years[np.isfinite(birth_years)]
        min_year = float(valid_birth.min()) if valid_birth.size else 1900.0
        max_year = float(valid_birth.max()) if valid_birth.size else 2024.0

        first_visit = {pid: data["dates"].min() for pid, data in patient_events.items()}
        all_pids = np.array(list(first_visit.keys()))
        all_fvs = np.array([first_visit[p] for p in all_pids])
        order = np.argsort(all_fvs)
        pids_sorted = all_pids[order]
        fv_sorted_dates = all_fvs[order]

        treat_code_set = set(self.treat_codes)
        out_code_set = set(self.out_codes)
        t0_trigger_set = set(t0_trigger_codes or self.treat_codes)

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

        print(f"Found {len(treated_cohort)} eligible treated patients. Sampling controls...")
        control_cohort: List[Dict] = []
        for tr in treated_cohort:
            t0 = tr["T0"]
            cutoff_np = t0 - np.timedelta64(int(baseline_window_days), "D")
            idx = np.searchsorted(fv_sorted_dates, cutoff_np, side="right")
            pool = pids_sorted[:idx]
            if len(pool) == 0:
                continue

            sampled_controls = []
            candidate_pool = np.random.choice(pool, size=min(2000, len(pool)), replace=False)
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

        full_cohort = treated_cohort + control_cohort
        self.n_treated = len(treated_cohort)
        self.n_controls = len(control_cohort)
        self.cohort_size = len(full_cohort)
        print(f"Final emulated cohort size: {self.cohort_size} (treated={self.n_treated}, controls={self.n_controls})")

        for row in full_cohort:
            pid = row["pid"]
            t0 = row["T0"].astype("datetime64[ns]")
            is_treated = int(row["T"])
            data = patient_events[pid]
            dates = data["dates"]
            hist_codes = data["hist_codes"]
            out_codes_arr = data["out_codes"]

            birth_year = data["birth_year"]
            norm_year = (birth_year - min_year) / (max_year - min_year + 1e-6) if np.isfinite(birth_year) else 0.5
            sex_num = float(data["sex_num"])
            t0_idx = np.searchsorted(dates, t0, side="right") - 1
            frailty_val = float(data["frailty"][t0_idx]) if t0_idx >= 0 else 0.0
            if not np.isfinite(frailty_val):
                frailty_val = 0.0
            norm_frailty = frailty_val / 10.0

            baseline_start = t0 - np.timedelta64(int(baseline_window_days), "D")
            b_mask = (dates >= baseline_start) & (dates <= t0)
            b_dates = dates[b_mask]
            b_hist_codes = hist_codes[b_mask]
            b_out_codes = out_codes_arr[b_mask]

            b_mask_strict = (dates >= baseline_start) & (dates < t0)
            b_out_codes_strict = out_codes_arr[b_mask_strict]
            historical_counts = np.zeros(self.n_outcomes, dtype=np.float32)
            for c in b_out_codes_strict:
                j = self.out2idx.get(c)
                if j is not None:
                    historical_counts[j] += 1.0
            norm_historical_counts = historical_counts / 20.0

            util_last30 = float(np.sum((dates >= (t0 - np.timedelta64(30, 'D'))) & (dates < t0))) / 10.0
            base_covariates = [norm_year, norm_frailty, sex_num] + norm_historical_counts.tolist() + [util_last30]

            f_start = t0 + np.timedelta64(int(buffer_days), 'D')
            f_end = f_start + np.timedelta64(int(followup_window_days), 'D')
            f_mask = (dates > f_start) & (dates <= f_end)
            f_out_codes = out_codes_arr[f_mask]

            p_end = t0 + np.timedelta64(int(buffer_days), 'D')
            p_mask = (dates > t0) & (dates <= p_end)
            p_out_codes = out_codes_arr[p_mask]

            final_followup_outcome = np.zeros(self.n_outcomes, dtype=np.float32)
            for ev in f_out_codes:
                j = self.out2idx.get(ev)
                if j is not None:
                    final_followup_outcome[j] = 1.0

            placebo_outcome = np.zeros(self.n_outcomes, dtype=np.float32)
            for ev in p_out_codes:
                j = self.out2idx.get(ev)
                if j is not None:
                    placebo_outcome[j] = 1.0

            unique_b_dates = np.unique(b_dates)
            covariates, treatments_seq, outcomes_seq = [], [], []
            for d in unique_b_dates:
                daily_mask = b_dates == d
                daily_hist = b_hist_codes[daily_mask]
                daily_out = b_out_codes[daily_mask]
                multi_hot_treat = np.zeros(self.n_treatments, dtype=np.float32)
                multi_hot_out = np.zeros(self.n_outcomes, dtype=np.float32)
                for ev in daily_hist:
                    i = self.treat2idx.get(ev)
                    if i is not None:
                        multi_hot_treat[i] = 1.0
                for ev in daily_out:
                    j = self.out2idx.get(ev)
                    if j is not None:
                        multi_hot_out[j] = 1.0
                treatments_seq.append(multi_hot_treat)
                outcomes_seq.append(multi_hot_out)
                covariates.append(base_covariates)

            if len(unique_b_dates) == 0 or unique_b_dates[-1] != t0:
                treatments_seq.append(np.zeros(self.n_treatments, dtype=np.float32))
                outcomes_seq.append(np.zeros(self.n_outcomes, dtype=np.float32))
                covariates.append(base_covariates)

            if is_treated:
                for tc in self.treat_codes:
                    idx_tc = self.treat2idx.get(tc)
                    if idx_tc is not None:
                        treatments_seq[-1][idx_tc] = 1.0

            treatments_seq.append(np.zeros(self.n_treatments, dtype=np.float32))
            outcomes_seq.append(final_followup_outcome)
            covariates.append(base_covariates)

            if len(treatments_seq) > self.max_seq_len:
                treatments_seq = treatments_seq[-self.max_seq_len:]
                outcomes_seq = outcomes_seq[-self.max_seq_len:]
                covariates = covariates[-self.max_seq_len:]

            self.patients.append({
                "pid": pid,
                "T": is_treated,
                "placebo": placebo_outcome,
                "covariates": np.asarray(covariates, dtype=np.float32),
                "treatments": np.asarray(treatments_seq, dtype=np.float32),
                "outcomes": np.asarray(outcomes_seq, dtype=np.float32),
                "seq_len": len(treatments_seq),
            })

        self.static_dim = 3
        self.dynamic_extra_dim = 1
        self.cov_dim = 3 + self.n_outcomes + 1

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pat = self.patients[idx]
        cov = torch.tensor(pat["covariates"], dtype=torch.float32)
        trt = torch.tensor(pat["treatments"], dtype=torch.float32)
        out = torch.tensor(pat["outcomes"], dtype=torch.float32)
        seq_len = int(pat["seq_len"])

        outputs = torch.zeros_like(out)
        outputs[:-1] = out[1:]
        prev_trt = torch.cat([torch.zeros(1, trt.shape[1]), trt[:-1]], dim=0)

        return {
            "pid": pat["pid"],
            "treated": torch.tensor(float(pat["T"]), dtype=torch.float32),
            "placebo_outcome": torch.tensor(pat["placebo"], dtype=torch.float32),
            "current_covariates": cov,
            "prev_treatments": prev_trt,
            "current_treatments": trt,
            "outputs": outputs,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(int(b["seq_len"].item()) for b in batch)
    cov_dim = batch[0]["current_covariates"].shape[1]
    n_treatments = batch[0]["current_treatments"].shape[1]
    n_outcomes = batch[0]["outputs"].shape[1]

    padded_cov = torch.zeros(len(batch), max_len, cov_dim)
    padded_prev_trt = torch.zeros(len(batch), max_len, n_treatments)
    padded_cur_trt = torch.zeros(len(batch), max_len, n_treatments)
    padded_out = torch.zeros(len(batch), max_len, n_outcomes)
    mask = torch.zeros(len(batch), max_len)

    seq_lens = torch.tensor([int(b["seq_len"].item()) for b in batch], dtype=torch.long)
    treated = torch.stack([b["treated"] for b in batch], dim=0)
    placebo = torch.stack([b["placebo_outcome"] for b in batch], dim=0)
    pids = [b["pid"] for b in batch]

    for i, b in enumerate(batch):
        slen = int(b["seq_len"].item())
        padded_cov[i, :slen] = b["current_covariates"]
        padded_prev_trt[i, :slen] = b["prev_treatments"]
        padded_cur_trt[i, :slen] = b["current_treatments"]
        padded_out[i, :slen] = b["outputs"]
        mask[i, :slen] = 1.0

    return {
        "pids": pids,
        "treated": treated,
        "placebo_outcome": placebo,
        "current_covariates": padded_cov,
        "prev_treatments": padded_prev_trt,
        "current_treatments": padded_cur_trt,
        "outputs": padded_out,
        "active_entries": mask.unsqueeze(-1),
        "seq_len": seq_lens,
    }


def build_weighted_sampler(dataset: Dataset, indices: Optional[Sequence[int]] = None):
    if indices is None:
        pats = dataset.patients
    else:
        pats = [dataset.patients[int(i)] for i in indices]
    labels = np.array([int(p["T"]) for p in pats], dtype=int)
    if labels.size == 0:
        return None
    counts = np.bincount(labels, minlength=2).astype(float)
    if np.any(counts == 0):
        return None
    class_w = 1.0 / counts
    sample_w = np.array([class_w[y] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    weighted_sampling: bool = False,
    num_workers: int = 0,
):
    sampler = build_weighted_sampler(dataset) if weighted_sampling else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_dataloader(
    csv_path: str,
    batch_size: int = 32,
    max_seq_len: int = 60,
    shuffle: bool = True,
    tte_args: Optional[dict] = None,
    num_workers: int = 0,
    weighted_sampling: bool = False,
):
    tte_args = tte_args or {}
    dataset = ChronosTargetTrialDataset(csv_path, max_seq_len=max_seq_len, **tte_args)
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        weighted_sampling=weighted_sampling,
        num_workers=num_workers,
    )
    return loader, dataset.n_treatments, dataset.n_outcomes, dataset.cov_dim
