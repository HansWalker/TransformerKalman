from __future__ import annotations
import sys
import torch
from Utils.initialize import import_from_config
from Training.train import train_models
from Eval.eval_models import evaluate, plot_results


def main():
    # single optional arg: path to root config
    root_cfg_path = sys.argv[1] if len(sys.argv) > 1 else "Configs/config.yaml"

    root_cfg, scene_import, model_imports, filter_imports = import_from_config(root_cfg_path)

    scenario_dim = root_cfg["Params"].get("dim", 3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(root_cfg["Params"].get("seed", 0))

    # Scenario: instantiate directly and read matrices from attributes
    scene_cls, scene_cfg = scene_import
    scene_params = scene_cfg.get("Params") or {}

    scenario = scene_cls(
        dimension=scenario_dim,
        device=device,
        dtype=dtype,
        **scene_params,
    )
    Q, R, P0, H = scenario.Q, scenario.R, scenario.P0, scenario.H
    state_dim = scenario.state_dim

    # Models (constructed up front; trained in-place)
    models = {}
    if root_cfg["Validation"].get("Load_Model"):
        model_paths = root_cfg["Validation"].get("Model_paths", {})
        # TODO: load models from paths into `models`
    else:
        for model_name, (model_cls, model_cfg) in model_imports.items():
            params = model_cfg.get("Params") or {}
            models[model_name] = model_cls(
                dim=state_dim,
                History=root_cfg["Params"].get("History", 5),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                **params,
            )

    # Filters
    filters = {}
    for filter_name, (filter_cls, filter_cfg) in filter_imports.items():
        params = filter_cfg.get("Params") or {}
        filters[filter_name] = filter_cls(
            scenario=scenario,
            Q=Q,
            R=R,
            P0=P0,
            **params,
        )

    # --- Train (if requested) ---
    if root_cfg["Training"].get("new_model", True) and models:
        train_models(
            models=list(models.values()),
            train_cfg=root_cfg["Training"],
            scenario=scenario,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    # --- Evaluate ---
    results = evaluate(
        models=models,
        filters=filters,
        scenario=scenario,
        val_cfg=root_cfg["Validation"],
        device=device,
        dtype=dtype,
    )
    plot_results(results, title="All methods vs ground truth")


if __name__ == "__main__":
    main()
