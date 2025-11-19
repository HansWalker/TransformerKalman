from __future__ import annotations
from typing import Dict, Any, Iterable, Optional, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

@torch.no_grad()
def evaluate(models: Dict[str, torch.nn.Module],
             filters: Dict[str, Any],
             scenario: Any,
             val_cfg: Dict[str, Any],
             device: torch.device,
             dtype: torch.dtype = torch.float32,
             models_ready_event: Optional[object] = None):
    """
    Computes mean per-time L2 errors to ground truth for each model/filter.

    Returns:
      {
        "time":   torch.Tensor of shape (T,), CPU
        "errors": { name: torch.Tensor of shape (T,), CPU }
      }
    """
    T = int(val_cfg.get("timesteps", 100))
    B = int(val_cfg.get("batch_size", 256))
    burn = int(val_cfg.get("burn_in", 0))  # optional
    workers = int(val_cfg.get("filter_workers", max(1, os.cpu_count() or 1)))

    # Allocate running sums on device
    names = list(models.keys()) + list(filters.keys())
    sums: Dict[str, torch.Tensor] = {n: torch.zeros(T, device=device, dtype=dtype) for n in names}

    # One synthetic batch from the scenario
    X_true, Y, x0 = scenario.sample_batch(T=T, B=B)
    X_true, Y, x0 = X_true.to(device), Y.to(device), x0.to(device)

    # --- Filters (parallel over different filters) ---
    def _run_filter(item):
        fname, fobj = item
        with tqdm(total=1, desc=f"Filter {fname}", leave=False) as pbar:
            Xhat, _ = fobj.run(Y, x0)
            pbar.update(1)
        err = (Xhat.to(device) - X_true).norm(dim=-1)
        return fname, err.sum(dim=0)
    if filters:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_filter, itm): itm[0] for itm in filters.items()}
            with tqdm(total=len(futures), desc="Filters", leave=False) as pbar:
                for fut in as_completed(futures):
                    fname, sum_vec = fut.result()
                    sums[fname] += sum_vec
                    pbar.update(1)

    # --- Models (wait for training if needed) ---
    if models and models_ready_event is not None:
        models_ready_event.wait()

    for name, m in tqdm(list(models.items()), desc="Models", leave=False):
        m.eval()
        Xhat = m(Y)                                    # (B,T,N)
        err = (Xhat.to(device) - X_true).norm(dim=-1)  # (B,T)
        sums[name] += err.sum(dim=0)

    # Compute per-time mean over series; apply optional burn-in
    denom = float(B)
    sl = slice(burn, None)
    time_full = torch.arange(T, device=device, dtype=dtype)[sl].cpu()
    errors = {}
    for n in names:
        err = (sums[n] / denom)[sl]
        finite_mask = torch.isfinite(err)
        trimmed_time = time_full[finite_mask.cpu().numpy().astype(bool)]
        err = err[finite_mask]
        errors[n] = (trimmed_time, err.cpu())

    return {"time": time_full, "errors": errors}


def plot_results(results, title: str | None = None, figsize=(10, 5)):
    """
    results can be either:
      {"time": (T,), "errors": {name: (T,), ...}}
    or just:
      {name: (T,), ...}
    """
    # normalize structure
    if isinstance(results, dict) and "errors" in results:
        errors = results["errors"]
        time_t = results.get("time", None)
    else:
        errors = results
        time_t = None

    # convert to numpy
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    series = {}
    prev_window = 20
    for name, data in errors.items():
        if isinstance(data, tuple):
            time_vals, err_vals = data
        else:
            time_vals = time_t if time_t is not None else torch.arange(len(data))
            err_vals = data

        if not isinstance(time_vals, torch.Tensor):
            time_vals = torch.tensor(time_vals)
        if not isinstance(err_vals, torch.Tensor):
            err_vals = torch.tensor(err_vals)

        invalid = torch.where((~torch.isfinite(err_vals)) | (err_vals > 10000.0))[0]
        if invalid.numel() > 0:
            cutoff = max(0, invalid[0].item() - prev_window)
            time_vals = time_vals[:cutoff]
            err_vals = err_vals[:cutoff]
        if err_vals.numel() == 0:
            continue
        series[name] = (_to_np(time_vals), _to_np(err_vals))
    if not series:
        return

    # plot (linear scale)
    plt.figure(figsize=figsize)
    for name, (t_vals, y_vals) in series.items():
        plt.plot(t_vals, y_vals, label=name, linewidth=2)

    plt.grid(True, alpha=0.3)
    plt.xlabel("time steps")
    plt.ylabel("mean L2 error to ground truth")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot (log10 scale)
    plt.figure(figsize=figsize)
    for name, (t_vals, y_vals) in series.items():
        y_vals = y_vals.copy()
        mask = y_vals > 0
        if not mask.any():
            continue
        plt.plot(t_vals[mask], np.log10(y_vals[mask]), label=name, linewidth=2)

    if plt.gca().has_data():
        plt.grid(True, alpha=0.3)
        plt.xlabel("time steps")
        plt.ylabel("log10 mean L2 error")
        if title:
            plt.title(f"{title} (log10)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
