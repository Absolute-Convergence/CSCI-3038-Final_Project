"""
worker.py

Behold!
This is the external PyTorch worker made for the Iris classification example.

This script accepts a learning_rate, hidden_size, epochs, and batch_size,
then writes validation_accuracy, validation_loss, and training_time_seconds
to the metrics file.

IMPORTANT NOTE!!! This file intentionally lives outside black_box_optimizer.
The optimizer should never ever import it and it shouldn't care how it
works internally. As far as the optimizer is concerned, this is just an
opaque subprocess that accepts parameters and eventually produces
metrics.

I wrote the model itself to remain simple in order best showcase how the
pipeline processes the (very tee-tiny) Iris dataset. That being said,
here are the design specs:

- one hidden layer using Linear -> ReLU -> Linear
- CrossEntropyLoss
- SGD
- an 80/20 training and validation split
- a fixed internal seed so identical hyperparameters give reproducible
  results during testing!
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

_DATA_PATH = Path(__file__).parent / "iris-data.csv"
_INPUT_SIZE = 4
_OUTPUT_SIZE = 3
_VALIDATION_FRACTION = 0.2
_INTERNAL_SEED = 0


def load_iris_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the included Iris data from disk"""
    features: list[list[float]] = []
    labels: list[int] = []

    with _DATA_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            features.append([
                float(row["sepal_length"]),
                float(row["sepal_width"]),
                float(row["petal_length"]),
                float(row["petal_width"]),
            ])
            labels.append(int(row["species"]))

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def split_train_validation(
    features: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shuffle the data predictably, then divide it into training/validation
    sets
    """
    generator = torch.Generator().manual_seed(_INTERNAL_SEED)
    permutation = torch.randperm(len(features), generator=generator)
    split_point = int(len(features) * (1.0 - _VALIDATION_FRACTION))

    train_idx = permutation[:split_point]
    val_idx = permutation[split_point:]

    return (
        features[train_idx],
        labels[train_idx],
        features[val_idx],
        labels[val_idx],
    )


class IrisClassifier(torch.nn.Module):
    """A tee-tiny feedforward network with only one hidden layer"""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_to_hidden = torch.nn.Linear(_INPUT_SIZE, hidden_size)
        self.activation = torch.nn.ReLU()
        self.hidden_to_output = torch.nn.Linear(hidden_size, _OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the hidden layer, activation, and output
        layer
        """
        x = self.input_to_hidden(x)
        x = self.activation(x)
        return self.hidden_to_output(x)


def train_and_evaluate(
    learning_rate: float, hidden_size: int, epochs: int, batch_size: int
) -> tuple[float, float]:
    """Train one model and return its validation accuracy and validation loss"""
    torch.manual_seed(_INTERNAL_SEED)

    features, labels = load_iris_data()
    train_x, train_y, val_x, val_y = split_train_validation(features, labels)

    model = IrisClassifier(hidden_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    model.train()

    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            # Must the gradients from the previous batch before calculating
            # the new ones
            optimizer.zero_grad()

            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)

            # Work backward from the loss then use those gradients to update
            # the model, classic
            loss.backward()
            optimizer.step()

    model.eval()

    # Evaluation does not update the model, so there is no reason to keep
    # tracking gradients here, duh
    with torch.no_grad():
        val_predictions = model(val_x)
        val_loss = loss_fn(val_predictions, val_y).item()
        val_accuracy = (
            (val_predictions.argmax(dim=-1) == val_y).float().mean().item()
        )

    return val_accuracy, val_loss


def write_metrics(
    metrics_path: Path,
    accuracy: float,
    loss: float,
    training_time_seconds: float,
) -> None:
    """Write one completed trial in the CSV format expected by metrics.py"""
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["validation_accuracy", "validation_loss", "training_time_seconds"]
        )
        writer.writerow([accuracy, loss, training_time_seconds])


def main() -> None:
    """
    Read the trial parameters, run the model, and write the resulting
    metrics.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--metrics-out", type=str, required=True)
    args = parser.parse_args()

    # Start timing immediately before training because we need to make
    # sure the resulting metric covers the full load, train, and
    # evaluation process

    start_time = time.perf_counter()

    accuracy, loss = train_and_evaluate(
        args.learning_rate,
        args.hidden_size,
        args.epochs,
        args.batch_size,
    )

    training_time_seconds = time.perf_counter() - start_time

    write_metrics(
        Path(args.metrics_out),
        accuracy,
        loss,
        training_time_seconds,
    )


if __name__ == "__main__":
    main()