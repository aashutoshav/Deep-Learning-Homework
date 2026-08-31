#!/usr/bin/env python3
"""
Training script for MNIST models.
Usage: python train_model.py --config <config_file> --run_name <run_name>
"""

import argparse
import os
import yaml
import copy
from tqdm import tqdm

from models import TwoLayerNet, SoftmaxRegression
from optimizer import SGD
from utils import (
    load_mnist_trainval,
    load_mnist_test,
    generate_batched_data,
    train,
    evaluate,
    plot_curves,
    write_training_log,
)


def train_model(
    config_file: str, run_name: str, overrides: dict = None
) -> tuple[list, list, list, list]:
    """
    Train a model using the specified configuration and save results.

    Args:
        config_file: Path to YAML configuration file
        run_name: Name for this training run (used for saving files)
        overrides: Optional dictionary of parameter overrides

    Returns:
        Tuple of (train_loss_history, train_acc_history, valid_loss_history, valid_acc_history)
    """
    # Load and flatten configuration
    with open(config_file) as f:
        config = yaml.full_load(f)

    args = {k: v for section in config.values() for k, v in section.items()}

    # Apply overrides
    if overrides:
        args.update({k: v for k, v in overrides.items() if v is not None})

    print(f"Starting training run: {run_name}")
    print(f"Configuration: {args}")

    # Prepare MNIST data
    train_data, train_label, val_data, val_label = load_mnist_trainval()
    test_data, test_label = load_mnist_test()

    # Prepare model and optimizer
    if args["type"] == "SoftmaxRegression":
        model = SoftmaxRegression()
        print("Using SoftmaxRegression model")
    elif args["type"] == "TwoLayerNet":
        model = TwoLayerNet(hidden_size=args["hidden_size"])
        print(f"Using TwoLayerNet model with hidden_size={args['hidden_size']}")
    else:
        raise ValueError(f"Unknown model type: {args['type']}")

    optimizer = SGD(learning_rate=args["learning_rate"], reg=args["reg"])
    print(f"Using SGD optimizer with lr={args['learning_rate']}, reg={args['reg']}")

    # Initialize training history and best model tracking
    train_loss_history, train_acc_history = [], []
    valid_loss_history, valid_acc_history = [], []
    best_acc, best_model = 0.0, None

    # Setup logging
    os.makedirs("plots", exist_ok=True)
    log_file_path = f"plots/{run_name}_detailed.log"

    print(f"\nStarting training for {args['epochs']} epochs...")

    with open(log_file_path, "w") as log_file:
        log_file.write(f"Training log for: {run_name}\nConfig: {args}\n\n")

        for epoch in tqdm(range(args["epochs"]), desc="Training"):
            # Training step
            batched_train_data, batched_train_label = generate_batched_data(
                train_data, train_label, batch_size=args["batch_size"], shuffle=True
            )
            epoch_loss, epoch_acc = train(
                epoch,
                batched_train_data,
                batched_train_label,
                model,
                optimizer,
                args["debug"],
                log_file,
            )
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            # Validation step
            batched_val_data, batched_val_label = generate_batched_data(
                val_data, val_label, batch_size=args["batch_size"]
            )
            valid_loss, valid_acc = evaluate(
                batched_val_data, batched_val_label, model, args["debug"], log_file
            )
            valid_loss_history.append(valid_loss)
            valid_acc_history.append(valid_acc)

            # Track best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = copy.deepcopy(model)

    # Test final model
    batched_test_data, batched_test_label = generate_batched_data(
        test_data, test_label, batch_size=args["batch_size"]
    )
    _, test_acc = evaluate(
        batched_test_data, batched_test_label, best_model, debug=False
    )

    print(
        f"\nTraining completed! Test Accuracy: {test_acc:.4f}, Best Val Accuracy: {best_acc:.4f}"
    )

    # Save results
    write_training_log(
        run_name,
        train_loss_history,
        train_acc_history,
        valid_loss_history,
        valid_acc_history,
        test_acc,
        best_acc,
        args,
    )
    plot_curves(
        train_loss_history,
        train_acc_history,
        valid_loss_history,
        valid_acc_history,
        run_name=run_name,
        save_only=True,
    )

    return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history


def main():
    parser = argparse.ArgumentParser(description="Train MNIST models")
    parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )
    parser.add_argument("--run_name", required=True, help="Name for this training run")
    parser.add_argument(
        "--learning_rate", "--lr", type=float, help="Override learning rate"
    )
    parser.add_argument(
        "--regularization", "--reg", type=float, help="Override regularization"
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found")
        return 1

    # Create overrides dictionary from non-None arguments
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in ["config", "run_name"] and v is not None
    }
    if "regularization" in overrides:
        overrides["reg"] = overrides.pop("regularization")

    try:
        train_model(args.config, args.run_name, overrides or None)
        return 0
    except Exception as e:
        print(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
