"""Train all 12 NSL-KDD configurations and all 64 seeds concurrently.

The job contains 12 x 64 = 768 independent CNNs in one grouped-convolution
model. A common microbatch of 32 keeps the full job within GPU memory. Gradient
accumulation preserves each configuration's effective batch size:

* batch 32: update after every microbatch
* batch 64: update after two microbatches
* batch 128: update after four microbatches

Each configuration owns a separate optimizer, so AdamW and coupled-L2 Adam
retain their distinct semantics even though their forward/backward work occurs
in the same CUDA graph.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import nsl_kdd_2dcnn as base
import nsl_kdd_gpu_sweep as sweep


MICROBATCH_SIZE = 32


def accumulation_steps(batch_size: int, microbatch_size: int = MICROBATCH_SIZE) -> int:
    if batch_size < microbatch_size or batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {batch_size} must be a positive multiple of "
            f"microbatch size {microbatch_size}."
        )
    return batch_size // microbatch_size


def accumulation_group_size(
    microbatch_index: int, total_microbatches: int, steps: int
) -> int:
    """Number of microbatches in the current optimizer-update group."""
    group_start = (microbatch_index // steps) * steps
    return min(steps, total_microbatches - group_start)


def optimizer_step_is_due(
    microbatch_index: int, total_microbatches: int, steps: int
) -> bool:
    return (microbatch_index + 1) % steps == 0 or (
        microbatch_index + 1 == total_microbatches
    )


def _torch():
    return sweep._torch()


class AllConcurrentCNN:
    """Factory for one module containing every configuration and seed replica."""

    @staticmethod
    def create(configs: list[sweep.SweepConfig], seeds: list[int], device: str):
        torch = _torch()
        nn = torch.nn
        functional = torch.nn.functional

        class _AllConcurrentCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config_count = len(configs)
                self.seed_count = len(seeds)
                # Reusing the same seed list gives controlled, identical initial
                # weights across configurations before their training choices diverge.
                self.config_models = nn.ModuleList(
                    [sweep.VectorizedCNN.create(seeds, "cpu") for _ in configs]
                )

            def forward(self, inputs):
                # [configuration, seed, batch, channel=1, height=6, width=6]
                config_count, seed_count, batch_size, _, height, width = inputs.shape
                if config_count != self.config_count or seed_count != self.seed_count:
                    raise ValueError(
                        "Input configuration/seed dimensions do not match the model."
                    )
                replica_count = config_count * seed_count
                values = inputs.reshape(replica_count, batch_size, 1, height, width)
                values = values.permute(1, 0, 2, 3, 4).reshape(
                    batch_size, replica_count, height, width
                )

                conv1_weight = torch.cat(
                    [model.conv1_weight for model in self.config_models], dim=0
                ).reshape(replica_count * 64, 1, 3, 3)
                conv1_bias = torch.cat(
                    [model.conv1_bias for model in self.config_models], dim=0
                ).reshape(-1)
                values = functional.conv2d(
                    values,
                    conv1_weight,
                    conv1_bias,
                    padding=1,
                    groups=replica_count,
                )
                values = functional.max_pool2d(
                    functional.relu(values), 2, stride=2, ceil_mode=True
                )

                conv2_weight = torch.cat(
                    [model.conv2_weight for model in self.config_models], dim=0
                ).reshape(replica_count * 32, 64, 3, 3)
                conv2_bias = torch.cat(
                    [model.conv2_bias for model in self.config_models], dim=0
                ).reshape(-1)
                values = functional.conv2d(
                    values,
                    conv2_weight,
                    conv2_bias,
                    padding=1,
                    groups=replica_count,
                )
                values = functional.max_pool2d(
                    functional.relu(values), 2, stride=2, ceil_mode=True
                )

                conv3_weight = torch.cat(
                    [model.conv3_weight for model in self.config_models], dim=0
                ).reshape(replica_count * 32, 32, 3, 3)
                conv3_bias = torch.cat(
                    [model.conv3_bias for model in self.config_models], dim=0
                ).reshape(-1)
                values = functional.conv2d(
                    values,
                    conv3_weight,
                    conv3_bias,
                    padding=1,
                    groups=replica_count,
                )
                values = functional.max_pool2d(
                    functional.relu(values), 2, stride=2, ceil_mode=True
                )

                values = values.reshape(batch_size, replica_count, 32).permute(1, 0, 2)
                dense_weight = torch.cat(
                    [model.dense_weight for model in self.config_models], dim=0
                )
                dense_bias = torch.cat(
                    [model.dense_bias for model in self.config_models], dim=0
                )
                logits = torch.einsum("rbi,roi->rbo", values, dense_weight)
                logits = logits + dense_bias[:, None, :]
                return logits.reshape(config_count, seed_count, batch_size, 5)

        return _AllConcurrentCNN().to(device)


def make_optimizers(
    model,
    configs: list[sweep.SweepConfig],
    learning_rate: float,
    weight_decay: float,
):
    torch = _torch()
    optimizers = []
    for config, config_model in zip(configs, model.config_models):
        optimizer_class = (
            torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
        )
        optimizers.append(
            optimizer_class(
                config_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        )
    return optimizers


def build_concurrent_inputs(
    feature_tensors: dict[str, Any],
    configs: list[sweep.SweepConfig],
    batch_indices,
):
    torch = _torch()
    return torch.stack(
        [feature_tensors[config.feature_mode][batch_indices] for config in configs],
        dim=0,
    )


def evaluate_all(
    model,
    configs: list[sweep.SweepConfig],
    feature_tensors: dict[str, Any],
    labels,
    indices,
    *,
    microbatch_size: int,
    use_bfloat16: bool,
) -> dict[str, np.ndarray]:
    torch = _torch()
    functional = torch.nn.functional
    config_count = len(configs)
    seed_count, row_count = indices.shape
    replica_count = config_count * seed_count
    loss_sum = torch.zeros(
        config_count, seed_count, device=labels.device, dtype=torch.float64
    )
    correct = torch.zeros(
        config_count, seed_count, device=labels.device, dtype=torch.int64
    )
    confusion = torch.zeros(replica_count * 25, device=labels.device, dtype=torch.int64)
    offsets = torch.arange(replica_count, device=labels.device).reshape(
        config_count, seed_count, 1
    ) * 25

    model.eval()
    with torch.inference_mode():
        for start in range(0, row_count, microbatch_size):
            batch_indices = indices[:, start : start + microbatch_size]
            batch_labels = labels[batch_indices]
            concurrent_inputs = build_concurrent_inputs(
                feature_tensors, configs, batch_indices
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                logits = model(concurrent_inputs)
                expanded_labels = batch_labels.unsqueeze(0).expand(config_count, -1, -1)
                losses = functional.cross_entropy(
                    logits.reshape(-1, 5),
                    expanded_labels.reshape(-1),
                    reduction="none",
                ).reshape(config_count, seed_count, -1)
            predictions = logits.argmax(dim=3)
            loss_sum += losses.double().sum(dim=2)
            correct += (predictions == expanded_labels).sum(dim=2)
            codes = offsets + expanded_labels * 5 + predictions
            confusion += torch.bincount(
                codes.reshape(-1), minlength=replica_count * 25
            )

    return {
        "loss": (loss_sum / row_count).cpu().numpy(),
        "accuracy": (correct.double() / row_count).cpu().numpy(),
        "confusion": confusion.reshape(config_count, seed_count, 5, 5).cpu().numpy(),
    }


def deterministic_epoch_shuffle(
    train_indices: np.ndarray, seeds: list[int], epoch: int
) -> np.ndarray:
    rows = []
    for replica, seed in enumerate(seeds):
        rng = np.random.default_rng(seed + 1009 + epoch * 1_000_003)
        rows.append(rng.permutation(train_indices[replica]))
    return np.stack(rows)


def save_checkpoint(
    path: Path,
    model,
    optimizers,
    configs: list[sweep.SweepConfig],
    seeds: list[int],
    epoch: int,
) -> None:
    torch = _torch()
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "configs": [asdict(config) for config in configs],
            "seeds": seeds,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: Path, model, optimizers, configs, seeds, device: str) -> int:
    torch = _torch()
    if not path.exists():
        return 1
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected_configs = [asdict(config) for config in configs]
    if checkpoint["configs"] != expected_configs or checkpoint["seeds"] != seeds:
        raise ValueError("Checkpoint configuration or seed list does not match this run.")
    model.load_state_dict(checkpoint["model_state_dict"])
    for optimizer, state in zip(optimizers, checkpoint["optimizer_state_dicts"]):
        optimizer.load_state_dict(state)
    return int(checkpoint["epoch"]) + 1


def train_all_concurrently(args: argparse.Namespace) -> None:
    torch = _torch()
    functional = torch.nn.functional
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the 768-model concurrent runner.")

    configs = sweep.experiment_matrix(
        args.batch_sizes, args.feature_modes, args.optimizers
    )
    seeds = sweep.select_random_seeds(args.seeds, args.master_seed)
    if len(configs) != 12 or len(seeds) != 64:
        raise ValueError(
            "The all-concurrent experiment requires exactly 12 configurations "
            "and 64 seeds (768 models)."
        )
    accumulation = [accumulation_steps(config.batch_size) for config in configs]

    source = args.data_dir / "KDDTrain+.txt"
    feature_data = {
        mode: sweep.prepare_features(source, mode) for mode in args.feature_modes
    }
    labels_np = next(iter(feature_data.values())).labels
    train_np, validation_np, test_np = sweep.build_seed_splits(labels_np, seeds)

    device = "cuda:0"
    torch.backends.cudnn.benchmark = True
    feature_tensors = {
        mode: torch.as_tensor(data.features, device=device)
        for mode, data in feature_data.items()
    }
    labels = torch.as_tensor(labels_np, device=device, dtype=torch.long)
    validation_indices = torch.as_tensor(validation_np, device=device, dtype=torch.long)
    test_indices = torch.as_tensor(test_np, device=device, dtype=torch.long)

    model = AllConcurrentCNN.create(configs, seeds, device)
    optimizers = make_optimizers(
        model, configs, args.learning_rate, args.weight_decay
    )
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint_latest.pt"
    history_path = args.output_dir / "all_histories.csv"
    start_epoch = load_checkpoint(
        checkpoint_path, model, optimizers, configs, seeds, device
    )
    if start_epoch > 1 and history_path.exists():
        existing_history = pd.read_csv(history_path)
        # If interruption happened after the CSV write but before the atomic
        # checkpoint replacement, discard the uncheckpointed epoch before retrying.
        history_rows = existing_history[
            existing_history["epoch"] < start_epoch
        ].to_dict("records")
    else:
        history_rows = []

    manifest = {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "configuration_count": len(configs),
        "seed_count": len(seeds),
        "concurrent_model_count": len(configs) * len(seeds),
        "microbatch_size": args.microbatch_size,
        "configs": [asdict(config) | {"config_id": config.config_id} for config in configs],
        "seeds": seeds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "precision": "bfloat16 autocast" if args.bfloat16 else "float32",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"{manifest['gpu']}: training all {manifest['concurrent_model_count']} "
        "models concurrently"
    )

    total_microbatches = math.ceil(train_np.shape[1] / args.microbatch_size)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        shuffled_np = deterministic_epoch_shuffle(train_np, seeds, epoch)
        shuffled = torch.as_tensor(shuffled_np, device=device, dtype=torch.long)
        train_loss_sum = torch.zeros(12, 64, device=device, dtype=torch.float64)
        train_correct = torch.zeros(12, 64, device=device, dtype=torch.int64)
        model.train()

        for microbatch_index, start in enumerate(
            range(0, shuffled.shape[1], args.microbatch_size)
        ):
            batch_indices = shuffled[:, start : start + args.microbatch_size]
            batch_labels = labels[batch_indices]
            concurrent_inputs = build_concurrent_inputs(
                feature_tensors, configs, batch_indices
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.bfloat16
            ):
                logits = model(concurrent_inputs)
                expanded_labels = batch_labels.unsqueeze(0).expand(12, -1, -1)
                per_item_loss = functional.cross_entropy(
                    logits.reshape(-1, 5),
                    expanded_labels.reshape(-1),
                    reduction="none",
                ).reshape(12, 64, -1)
                per_replica_loss = per_item_loss.mean(dim=2)
                scaled_losses = []
                for config_index, steps in enumerate(accumulation):
                    group_size = accumulation_group_size(
                        microbatch_index, total_microbatches, steps
                    )
                    scaled_losses.append(
                        per_replica_loss[config_index].sum() / group_size
                    )
                loss = torch.stack(scaled_losses).sum()
            loss.backward()

            for config_index, (optimizer, steps) in enumerate(
                zip(optimizers, accumulation)
            ):
                if optimizer_step_is_due(
                    microbatch_index, total_microbatches, steps
                ):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            train_loss_sum += per_item_loss.detach().double().sum(dim=2)
            train_correct += (
                logits.detach().argmax(dim=3) == expanded_labels
            ).sum(dim=2)

        validation = evaluate_all(
            model,
            configs,
            feature_tensors,
            labels,
            validation_indices,
            microbatch_size=args.microbatch_size,
            use_bfloat16=args.bfloat16,
        )
        train_losses = (train_loss_sum / shuffled.shape[1]).cpu().numpy()
        train_accuracies = (
            train_correct.double() / shuffled.shape[1]
        ).cpu().numpy()
        for config_index, config in enumerate(configs):
            for seed_index, seed in enumerate(seeds):
                history_rows.append(
                    {
                        "config_id": config.config_id,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": float(train_losses[config_index, seed_index]),
                        "train_accuracy": float(
                            train_accuracies[config_index, seed_index]
                        ),
                        "validation_loss": float(
                            validation["loss"][config_index, seed_index]
                        ),
                        "validation_accuracy": float(
                            validation["accuracy"][config_index, seed_index]
                        ),
                    }
                )
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        save_checkpoint(
            checkpoint_path, model, optimizers, configs, seeds, epoch
        )
        print(
            f"epoch {epoch:02d}/{args.epochs}: "
            f"mean_train_accuracy={train_accuracies.mean():.4f}, "
            f"mean_validation_accuracy={validation['accuracy'].mean():.4f}"
        )

    test = evaluate_all(
        model,
        configs,
        feature_tensors,
        labels,
        test_indices,
        microbatch_size=args.microbatch_size,
        use_bfloat16=args.bfloat16,
    )
    flattened_metrics = sweep.metrics_from_confusions(
        test["confusion"].reshape(12 * 64, 5, 5)
    )
    final_history = pd.DataFrame(history_rows)
    final_history = final_history[final_history["epoch"] == args.epochs]
    final_history = final_history.set_index(["config_id", "seed"])

    rows = []
    flat_index = 0
    for config_index, config in enumerate(configs):
        for seed_index, seed in enumerate(seeds):
            row = {
                "config_id": config.config_id,
                **asdict(config),
                "seed": seed,
                "validation_accuracy": float(
                    final_history.loc[(config.config_id, seed), "validation_accuracy"]
                ),
                "validation_loss": float(
                    final_history.loc[(config.config_id, seed), "validation_loss"]
                ),
                "test_accuracy": float(test["accuracy"][config_index, seed_index]),
                "test_loss": float(test["loss"][config_index, seed_index]),
            }
            row.update(
                {
                    name: float(values[flat_index])
                    for name, values in flattened_metrics.items()
                }
            )
            rows.append(row)
            flat_index += 1

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "all_seed_results.csv", index=False)
    summary = (
        results.groupby(
            ["config_id", "batch_size", "feature_mode", "optimizer"],
            as_index=False,
        )
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_accuracy_min=("test_accuracy", "min"),
            test_accuracy_max=("test_accuracy", "max"),
            macro_f1_mean=("macro_f1", "mean"),
        )
        .sort_values("test_accuracy_mean", ascending=False)
    )
    summary["test_accuracy_95ci_half_width"] = (
        1.96 * summary["test_accuracy_std"] / math.sqrt(64)
    )
    summary["total_elapsed_seconds"] = time.perf_counter() - started
    summary.to_csv(args.output_dir / "configuration_summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 12 configurations x 64 seeds concurrently on one GPU."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Datasets/NSL-KDD"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/nsl_kdd_all_768_concurrent"),
    )
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--master-seed", type=int, default=20260908)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--microbatch-size", type=int, default=MICROBATCH_SIZE)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(sweep.DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=sweep.DEFAULT_FEATURE_MODES,
        default=list(sweep.DEFAULT_FEATURE_MODES),
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=sweep.DEFAULT_OPTIMIZERS,
        default=list(sweep.DEFAULT_OPTIMIZERS),
    )
    parser.add_argument(
        "--float32",
        dest="bfloat16",
        action="store_false",
        help="Disable bfloat16 autocast.",
    )
    parser.set_defaults(bfloat16=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.microbatch_size != MICROBATCH_SIZE:
        raise ValueError(
            f"The verified 32/64/128 schedule requires microbatch size {MICROBATCH_SIZE}."
        )
    train_all_concurrently(args)


if __name__ == "__main__":
    main()
