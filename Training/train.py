import torch
import torch.nn as nn
from tqdm.auto import tqdm

BATCH_MULTIPLIER = 200  # pool mini-batches for data efficiency


# ---------- Gradient utilities (AMP- and vmap-safe) ----------

@torch.no_grad()
def sanitize_grads_(models: list[nn.Module]) -> dict:
    """
    Replace only the non-finite entries in each grad tensor; leave finite entries as-is.
    Returns a small dict of counts for logging.
    """
    stats = {"sanitized_tensors": 0, "sanitized_elems": 0}
    for m in models:
        for p in m.parameters():
            if p.grad is None:
                continue
            g = p.grad
            bad = ~torch.isfinite(g)
            if bad.any():
                # Replace NaN->0, +Inf/-Inf->clamped to 0 (keeps direction elsewhere)
                torch.nan_to_num_(g, nan=0.0, posinf=0.0, neginf=0.0)
                stats["sanitized_tensors"] += 1
                stats["sanitized_elems"] += int(bad.sum())
    return stats


@torch.no_grad()
def agc_(models: list[nn.Module], clip_factor: float = 0.01, eps: float = 1e-3):
    """
    Adaptive Gradient Clipping (unit-wise), following NFNets:
      For each param tensor W with grad G, reshape to (units, -1).
      If ||G_i|| > clip * (||W_i|| + eps), scale G_i down.
    Ref: Brock et al., "High-Performance Large-Scale Image Recognition Without Normalization"
    (AGC) :contentReference[oaicite:4]{index=4}
    """
    for m in models:
        for p in m.parameters():
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim <= 1:
                # Bias / vectors: treat as one unit
                p_norm = p.norm()
                g_norm = g.norm()
                max_norm = clip_factor * (p_norm + eps)
                if g_norm > max_norm:
                    p.grad.mul_(max_norm / (g_norm + 1e-12))
            else:
                # Unit-wise along the first dim (out-features / out-channels)
                # Flatten each unit to a vector
                g_flat = g.view(g.shape[0], -1)
                p_flat = p.view(p.shape[0], -1)
                g_norm = torch.linalg.norm(g_flat, dim=1, keepdim=True)         # (units,1)
                p_norm = torch.linalg.norm(p_flat, dim=1, keepdim=True)         # (units,1)
                max_norm = clip_factor * (p_norm + eps)
                # Scale factors in [0,1], applied only where g_norm > max_norm
                scale = (max_norm / (g_norm + 1e-12)).clamp(max=1.0)
                # Reshape back and scale grad in-place
                g.mul_(scale.view([-1] + [1] * (g.ndim - 1)))


@torch.no_grad()
def clip_grad_global_norm_(models: list[nn.Module], max_norm: float = 1.0, norm_type: float = 2.0):
    """Light global-norm clip as a backstop (PyTorch doc semantics). :contentReference[oaicite:5]{index=5}"""
    for m in models:
        torch.nn.utils.clip_grad_norm_(
            m.parameters(), max_norm=max_norm, norm_type=norm_type, error_if_nonfinite=False
        )


# ---------- Streamlined trainer with smart non-finite handling + AGC ----------

def train_models(models, train_cfg, scenario, device):
    """
    models: nn.Module or iterable of nn.Module (trained in lockstep on the same data)
    Strategy:
      1) AMP forward/backward (if available)
      2) Unscale grads
      3) Sanitize only bad entries, apply AGC (unit-wise), then small global-norm clip
      4) Step (skip only if everything still non-finite)
    """
    if isinstance(models, nn.Module):
        models = [models]
    for m in models:
        m.train().to(device)

    steps      = int(train_cfg.get("steps", 1000))
    lr         = float(train_cfg.get("lr", 1e-3))
    batch_size = int(train_cfg.get("batch_size", 64))
    timesteps  = int(train_cfg.get("timesteps", 100))

    # Fixed, always-on controls (no combinatorics)
    agc_clip   = float(train_cfg.get("agc_clip", 0.01))  # NFNets default-ish scale
    max_gnorm  = float(train_cfg.get("max_grad_norm", 1.0))
    gnorm_type = float(train_cfg.get("grad_norm_type", 2.0))

    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9) for opt in optimizers]
    loss_fn    = nn.MSELoss()

    # AMP setup (cuda and cpu backends). GradScaler does dynamic loss scaling and step skipping. :contentReference[oaicite:6]{index=6}
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        autocast_cm = torch.amp.autocast("cuda", enabled=True)
    else:
        scaler = torch.amp.GradScaler("cpu", enabled=True)
        autocast_cm = torch.amp.autocast("cpu", enabled=True)

    pool_B = BATCH_MULTIPLIER * batch_size
    X_pool = Y_pool = None

    for i in tqdm(range(steps), desc="Training", leave=False):
        j = i % BATCH_MULTIPLIER
        if j == 0:
            with torch.no_grad():
                X_pool, Y_pool, _ = scenario.sample_batch(T=timesteps, B=pool_B)
                X_pool = X_pool.to(device, non_blocking=True)
                Y_pool = Y_pool.to(device, non_blocking=True)

        s, e = j * batch_size, (j + 1) * batch_size
        X_mb, Y_mb = X_pool[s:e], Y_pool[s:e]

        # zero grads
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # forward/backward
        with autocast_cm:
            losses = [loss_fn(m(Y_mb), X_mb) for m in models]
            total_loss = torch.stack(losses).mean()
        scaler.scale(total_loss).backward()

        # Unscale then sanitize -> AGC -> small global-norm clip
        for opt in optimizers:
            scaler.unscale_(opt)

        stats = sanitize_grads_(models)               # only bad entries are fixed
        agc_(models, clip_factor=agc_clip)            # unit-wise, per NFNets :contentReference[oaicite:7]{index=7}
        clip_grad_global_norm_(models, max_norm=max_gnorm, norm_type=gnorm_type)  # safety net :contentReference[oaicite:8]{index=8}

        # If (after sanitize+AGC+clip) any grad tensor is still entirely non-finite, skip this step
        # (rare; GradScaler would also skip on overflow) :contentReference[oaicite:9]{index=9}
        any_all_bad = False
        for m in models:
            for p in m.parameters():
                if p.grad is None:
                    continue
                # If ALL entries in this grad are non-finite, it’s unrecoverable for this param
                if (~torch.isfinite(p.grad)).all():
                    any_all_bad = True
                    break
            if any_all_bad:
                break

        if any_all_bad:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            scaler.update()  # let AMP lower the loss scale automatically :contentReference[oaicite:10]{index=10}
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Step {i+1}/{steps}  skipped (unrecoverable non-finite grads)")
            continue

        # optimizer step (AMP)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()

        # (optional) schedule
        if (i + 1) % 100 == 0 and (i + 1) > 1000:
            for sch in schedulers:
                sch.step()
