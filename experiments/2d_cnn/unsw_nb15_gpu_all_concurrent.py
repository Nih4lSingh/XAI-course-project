"""Train all paper-relevant UNSW-NB15 configurations concurrently on CUDA.

The default matrix covers batch sizes 32, 64, and 128 plus both plausible
weight-decay semantics: decoupled AdamW and Adam with coupled L2. This gives
6 configurations x 64 seeds = 384 independent CNNs in one GPU job.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import nsl_kdd_gpu_all_concurrent as shared_all
import unsw_nb15_gpu_sweep as sweep


MICROBATCH_SIZE = 32
accumulation_steps = shared_all.accumulation_steps
accumulation_group_size = shared_all.accumulation_group_size
optimizer_step_is_due = shared_all.optimizer_step_is_due
deterministic_epoch_shuffle = shared_all.deterministic_epoch_shuffle
AllConcurrentCNN = shared_all.AllConcurrentCNN
make_optimizers = shared_all.make_optimizers
save_checkpoint = shared_all.save_checkpoint
load_checkpoint = shared_all.load_checkpoint


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class TerminalProgress:
    """Small dependency-free progress bar suitable for Windows PowerShell."""

    def __init__(self, total: int, initial: int = 0, width: int = 32) -> None:
        self.total = total
        self.initial = initial
        self.width = width
        self.started = time.perf_counter()
        self.draw_every = max(1, total // 200)
        self.last_drawn = initial - self.draw_every

    def update(
        self,
        completed: int,
        *,
        epoch: int,
        epochs: int,
        batch: int,
        batches: int,
        force: bool = False,
    ) -> None:
        # Redrawing about 200 times per run keeps the display responsive without
        # adding measurable overhead to thousands of CUDA microbatches.
        if (
            not force
            and completed < self.total
            and completed - self.last_drawn < self.draw_every
        ):
            return
        self.last_drawn = completed
        fraction = min(1.0, completed / self.total) if self.total else 1.0
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.started
        work_done = max(0, completed - self.initial)
        remaining = max(0, self.total - completed)
        eta = elapsed * remaining / work_done if work_done else 0.0
        message = (
            f"\r[{bar}] {fraction * 100:6.2f}%  "
            f"epoch {epoch:02d}/{epochs}  batch {batch:04d}/{batches:04d}  "
            f"elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}"
        )
        print(message, end="", flush=True)

    def stage(self, label: str) -> None:
        print(f"\n{label}", flush=True)

    def finish(self) -> None:
        print(flush=True)


def _torch():
    return sweep._torch()


def build_concurrent_inputs(features, configs: list[sweep.SweepConfig], batch_indices):
    """Return [configuration, seed, batch, channel, 7, 7] inputs."""
    per_seed = features[batch_indices]
    return per_seed.unsqueeze(0).expand(len(configs), *per_seed.shape)


def evaluate_all(
    model,
    configs: list[sweep.SweepConfig],
    features,
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
            inputs = build_concurrent_inputs(features, configs, batch_indices)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                logits = model(inputs)
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


def train_all_concurrently(args: argparse.Namespace) -> None:
    torch = _torch()
    functional = torch.nn.functional
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the UNSW-NB15 concurrent runner.")

    configs = sweep.experiment_matrix(args.batch_sizes, args.optimizers)
    seeds = sweep.select_random_seeds(args.seeds, args.master_seed)
    if len(configs) != 6 or len(seeds) != 64:
        raise ValueError(
            "The all-concurrent experiment requires exactly 6 configurations "
            "and 64 seeds (384 models)."
        )
    config_count = len(configs)
    seed_count = len(seeds)
    accumulation = [accumulation_steps(config.batch_size) for config in configs]

    feature_data = sweep.prepare_features(args.data_dir, args.sample_seed)
    train_np, validation_np, test_np = sweep.build_seed_splits(
        feature_data.labels, seeds
    )

    device = "cuda:0"
    torch.backends.cudnn.benchmark = True
    features = torch.as_tensor(feature_data.features, device=device)
    labels = torch.as_tensor(feature_data.labels, device=device, dtype=torch.long)
    validation_indices = torch.as_tensor(validation_np, device=device, dtype=torch.long)
    test_indices = torch.as_tensor(test_np, device=device, dtype=torch.long)

    model = AllConcurrentCNN.create(configs, seeds, device)
    optimizers = make_optimizers(model, configs, args.learning_rate, args.weight_decay)
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint_latest.pt"
    history_path = args.output_dir / "all_histories.csv"
    manifest_path = args.output_dir / "manifest.json"
    if checkpoint_path.exists():
        if not manifest_path.exists():
            raise ValueError(
                "A checkpoint exists without its manifest; use a new output directory."
            )
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(previous_manifest.get("sample_seed", -1)) != args.sample_seed:
            raise ValueError(
                "The existing checkpoint used a different sample seed; use a new "
                "output directory instead of mixing two sampled datasets."
            )
    start_epoch = load_checkpoint(
        checkpoint_path, model, optimizers, configs, seeds, device
    )
    if start_epoch > 1 and history_path.exists():
        existing_history = pd.read_csv(history_path)
        history_rows = existing_history[
            existing_history["epoch"] < start_epoch
        ].to_dict("records")
    else:
        history_rows = []

    manifest = {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "configuration_count": config_count,
        "seed_count": seed_count,
        "concurrent_model_count": config_count * seed_count,
        "microbatch_size": args.microbatch_size,
        "configs": [asdict(config) | {"config_id": config.config_id} for config in configs],
        "seeds": seeds,
        "sample_seed": args.sample_seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "precision": "bfloat16 autocast" if args.bfloat16 else "float32",
        "paper_targets": {"test_accuracy": 0.81, "test_loss": 0.40},
        "data": feature_data.metadata,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"{manifest['gpu']}: training all {manifest['concurrent_model_count']} "
        "models concurrently"
    )
    print(
        f"UNSW-NB15 paper sample: {len(feature_data.labels)} rows, "
        f"train/validation/test={train_np.shape[1]}/{validation_np.shape[1]}/{test_np.shape[1]}"
    )

    total_microbatches = math.ceil(train_np.shape[1] / args.microbatch_size)
    total_training_steps = args.epochs * total_microbatches
    completed_before_resume = max(0, start_epoch - 1) * total_microbatches
    progress = TerminalProgress(total_training_steps, completed_before_resume)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        shuffled_np = deterministic_epoch_shuffle(train_np, seeds, epoch)
        shuffled = torch.as_tensor(shuffled_np, device=device, dtype=torch.long)
        train_loss_sum = torch.zeros(
            config_count, seed_count, device=device, dtype=torch.float64
        )
        train_correct = torch.zeros(
            config_count, seed_count, device=device, dtype=torch.int64
        )
        model.train()

        for microbatch_index, start in enumerate(
            range(0, shuffled.shape[1], args.microbatch_size)
        ):
            batch_indices = shuffled[:, start : start + args.microbatch_size]
            batch_labels = labels[batch_indices]
            inputs = build_concurrent_inputs(features, configs, batch_indices)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.bfloat16
            ):
                logits = model(inputs)
                expanded_labels = batch_labels.unsqueeze(0).expand(config_count, -1, -1)
                per_item_loss = functional.cross_entropy(
                    logits.reshape(-1, 5),
                    expanded_labels.reshape(-1),
                    reduction="none",
                ).reshape(config_count, seed_count, -1)
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

            for optimizer, steps in zip(optimizers, accumulation):
                if optimizer_step_is_due(
                    microbatch_index, total_microbatches, steps
                ):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            train_loss_sum += per_item_loss.detach().double().sum(dim=2)
            train_correct += (
                logits.detach().argmax(dim=3) == expanded_labels
            ).sum(dim=2)
            completed_steps = (epoch - 1) * total_microbatches + microbatch_index + 1
            progress.update(
                completed_steps,
                epoch=epoch,
                epochs=args.epochs,
                batch=microbatch_index + 1,
                batches=total_microbatches,
                force=microbatch_index + 1 == total_microbatches,
            )

        progress.stage(f"epoch {epoch:02d}/{args.epochs}: validating all models")
        validation = evaluate_all(
            model,
            configs,
            features,
            labels,
            validation_indices,
            microbatch_size=args.microbatch_size,
            use_bfloat16=args.bfloat16,
        )
        train_losses = (train_loss_sum / shuffled.shape[1]).cpu().numpy()
        train_accuracies = (train_correct.double() / shuffled.shape[1]).cpu().numpy()
        for config_index, config in enumerate(configs):
            for seed_index, seed in enumerate(seeds):
                history_rows.append(
                    {
                        "config_id": config.config_id,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": float(train_losses[config_index, seed_index]),
                        "train_accuracy": float(train_accuracies[config_index, seed_index]),
                        "validation_loss": float(validation["loss"][config_index, seed_index]),
                        "validation_accuracy": float(
                            validation["accuracy"][config_index, seed_index]
                        ),
                    }
                )
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        save_checkpoint(checkpoint_path, model, optimizers, configs, seeds, epoch)
        print(
            f"epoch {epoch:02d}/{args.epochs}: "
            f"mean_train_accuracy={train_accuracies.mean():.4f}, "
            f"mean_validation_accuracy={validation['accuracy'].mean():.4f}"
        )

    progress.finish()
    print(
        f"Evaluating all {config_count * seed_count} models on their test splits",
        flush=True,
    )
    test = evaluate_all(
        model,
        configs,
        features,
        labels,
        test_indices,
        microbatch_size=args.microbatch_size,
        use_bfloat16=args.bfloat16,
    )
    flattened_metrics = sweep.metrics_from_confusions(
        test["confusion"].reshape(config_count * seed_count, 5, 5)
    )
    final_history = pd.DataFrame(history_rows)
    final_history = final_history[final_history["epoch"] == args.epochs]
    final_history = final_history.set_index(["config_id", "seed"])

    rows: list[dict[str, Any]] = []
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
        results.groupby(["config_id", "batch_size", "optimizer"], as_index=False)
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
        1.96 * summary["test_accuracy_std"] / math.sqrt(seed_count)
    )
    summary["total_elapsed_seconds"] = time.perf_counter() - started
    summary.to_csv(args.output_dir / "configuration_summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 6 UNSW-NB15 configurations x 64 seeds concurrently."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Datasets/UNSW-NB15"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/unsw_nb15_all_384_concurrent"),
    )
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--master-seed", type=int, default=20260909)
    parser.add_argument("--sample-seed", type=int, default=20260909)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--microbatch-size", type=int, default=MICROBATCH_SIZE)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(sweep.DEFAULT_BATCH_SIZES)
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
