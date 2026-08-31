import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from jaxtyping import Float, Int


def load_csv(path):
    """
    Load the CSV form of MNIST data without any external library
    :param path: the path of the csv file
    :return:
        data: A list of list where each sub-list with 28x28 elements
              corresponding to the pixels in each image
        labels: A list containing labels of images
    """
    data = []
    labels = []
    with open(path, "r") as fp:
        images = fp.readlines()
        images = [img.rstrip() for img in images]

        for img in images:
            img_as_list = img.split(",")
            y = int(img_as_list[0])  # first entry as label
            x = img_as_list[1:]
            x = [int(px) / 255 for px in x]
            data.append(x)
            labels.append(y)
    return data, labels


def load_mnist_trainval():
    """
    Load MNIST training data with labels
        1) Split mnist_train.csv into training data and validation data.
        2) Use the first 80% for training and the last 20% for validation.
        3) You do not need to shuffle the data.

    :return:
        train_data: A list of lists containing the training data, each inner list contains 28x28 elements corresponding to pixel values in images
        train_label: A list containing the labels of training data
        test_data: A list of lists containing the testing data, each inner list contains 28x28 elements corresponding to pixel values in images
        test_label: A list containing the labels of testing data
    """
    # Load training data
    print("Loading training data...")
    data, label = load_csv("./data/mnist_train.csv")
    assert len(data) == len(label)
    print("Training data loaded with {count} images".format(count=len(data)))

    # split training/validation data
    train_data = None
    train_label = None
    val_data = None
    val_label = None

    ### TODO: BEGIN SOLUTION ###
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    train_label = label[:split_idx]
    val_data = data[split_idx:]
    val_label = label[split_idx:]
    ### END SOLUTION ###

    return train_data, train_label, val_data, val_label


def load_mnist_test():
    """
    Load MNIST testing data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        test_data: A list of list containing the testing data
        test_label: A list containing the labels of testing data
    """
    # Load training data
    print("Loading testing data...")
    data, label = load_csv("./data/mnist_test.csv")
    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    return data, label


def generate_batched_data(
    data: list[list[float]],
    label: list[int],
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int | None = None,
) -> tuple[
    list[Float[np.ndarray, "batch_size 784"]], list[Int[np.ndarray, "batch_size"]]
]:
    """
    Turn raw data into batched forms
        1) Shuffle data using random.shuffle and label if shuffle=True
        2) Generate batches of images with the required batch size
           It's okay if the size of your last batch is smaller than the required
           batch size

        Important:
        1) The individual batches must be NumPy arrays.
        2) Use random.shuffle and not np.random.shuffle
        3) Shuffle the data and label in the same way

    :param data: A list of lists where each inner list contains 28x28
                 elements corresponding to pixel values in images: [[pix1, ..., pix768], ..., [pix1, ..., pix768]]
    :param label: A list containing the labels of data
    :param batch_size: required batch size
    :param shuffle: Whether to shuffle the data: true for training and False for testing
    :param seed: Optional seed for random number generator
    :return:
        batched_data: A list of NumPy arrays representing batches of images. The NumPy arrays should be of shape (batch_size, 784) except for the last batch.
        batched_label: A list of NumPy arrays representing batches of labels. The NumPy arrays should be of shape (batch_size,)
    """
    batched_data = None
    batched_label = None
    if seed is not None:
        random.seed(seed)
    ### TODO: BEGIN SOLUTION ###
    if shuffle:
        combined = list(zip(data, label))
        random.shuffle(combined)
        shuffled_data, shuffled_label = zip(*combined)
        data = list(shuffled_data)
        label = list(shuffled_label)

    batched_data = []
    batched_label = []
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        batch_x = np.array(data[i : i + batch_size], dtype=float)
        batch_y = np.array(label[i : i + batch_size], dtype=int)
        batched_data.append(batch_x)
        batched_label.append(batch_y)
    ### END SOLUTION ###

    return batched_data, batched_label


def train(
    epoch,
    batched_train_data,
    batched_train_label,
    model,
    optimizer,
    debug=True,
    log_file=None,
):
    """
    A training function that trains the model for one epoch
    :param epoch: The index of current epoch
    :param batched_train_data: A list containing batches of images
    :param batched_train_label: A list containing batches of labels
    :param model: The model to be trained
    :param optimizer: The optimizer that updates the network weights
    :param debug: Whether to log detailed batch information
    :param log_file: Optional file handle to write logs to instead of console
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_train_data, batched_train_label)):
        start_time = time.time()
        loss, accuracy = model.forward(input, target)

        optimizer.update(model)
        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]

        if idx % 10 == 0 and debug:
            log_message = f"Epoch: [{epoch}][{idx}/{len(batched_train_data)}]\tBatch Loss {loss:.4f}\tTrain Accuracy {accuracy:.4f}"
            if log_file:
                log_file.write(log_message + "\n")
                log_file.flush()
            else:
                print(log_message)
    epoch_loss /= len(batched_train_data)
    epoch_acc = hits / count_samples

    if debug:
        log_message = f"* Average Accuracy of Epoch {epoch} is: {epoch_acc:.4f}"
        if log_file:
            log_file.write(log_message + "\n")
            log_file.flush()
        else:
            print(log_message)
    return epoch_loss, epoch_acc


def evaluate(batched_test_data, batched_test_label, model, debug=True, log_file=None):
    """
    Evaluate the model on test data
    :param batched_test_data: A list containing batches of test images
    :param batched_test_label: A list containing batches of labels
    :param model: A pre-trained model
    :param debug: Whether to log detailed batch information
    :param log_file: Optional file handle to write logs to instead of console
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):
        loss, accuracy = model.forward(input, target, mode="valid")

        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]
        if debug:
            log_message = f"Evaluate: [{idx}/{len(batched_test_data)}]\tBatch Accuracy {accuracy:.4f}"
            if log_file:
                log_file.write(log_message + "\n")
                log_file.flush()
            else:
                print(log_message)
    epoch_loss /= len(batched_test_data)
    epoch_acc = hits / count_samples

    return epoch_loss, epoch_acc


def plot_curves(
    train_loss_history,
    train_acc_history,
    valid_loss_history,
    valid_acc_history,
    run_name=None,
    save_only=False,
):
    """
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
        1) Plot learning curves of training and validation loss
        2) Plot learning curves of training and validation accuracy

    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :param run_name: optional name for the run, used for saving plots and title
    :param save_only: if True, only save plots without displaying them
    :return: None, save two figures in the current directory
    """
    epochs = range(len(train_loss_history))
    os.makedirs("plots", exist_ok=True)

    # Helper function to create and save plot
    def create_plot(train_data, valid_data, ylabel, filename_suffix):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_data, label="train", marker="o")
        plt.plot(epochs, valid_data, label="valid", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(f"{ylabel} Curve - {run_name}" if run_name else f"{ylabel} Curve")
        plt.grid(True, alpha=0.3)

        if run_name:
            filename = f"plots/{run_name}_{filename_suffix}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"{ylabel} plot saved to: {filename}")

        if not save_only:
            plt.show()
        plt.close()

    create_plot(train_loss_history, valid_loss_history, "Loss", "loss")
    create_plot(train_acc_history, valid_acc_history, "Accuracy", "accuracy")


def load_training_logs(run_name):
    """
    Load training results from a log file.

    :param run_name: Name of the training run
    :return: Dictionary containing training results
    """
    import json

    log_file = f"plots/{run_name}.log"
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    with open(log_file, "r") as f:
        for line in f:
            if line.startswith("RESULTS:"):
                return json.loads(line[8:])  # Remove 'RESULTS:' prefix

    raise ValueError(f"No results found in log file: {log_file}")


def display_training_plots(run_name):
    """
    Display the saved plots for a training run in a notebook.

    :param run_name: Name of the training run
    """
    from IPython.display import Image, display

    # Display summary from log file
    try:
        results = load_training_logs(run_name)
        config = results["config"]
        print(f"Run: {run_name}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Best Validation Accuracy: {results['best_validation_accuracy']:.4f}")
        print(
            f"Model: {config['type']}, LR: {config['learning_rate']}, Reg: {config['reg']}"
        )
        print("-" * 50)
    except (FileNotFoundError, ValueError) as e:
        print(f"Could not load training summary: {e}")

    # Display plots
    for plot_type in ["loss", "accuracy"]:
        plot_file = f"plots/{run_name}_{plot_type}.png"
        if os.path.exists(plot_file):
            print(f"{plot_type.title()} Curve:")
            display(Image(plot_file))
        else:
            print(f"{plot_type.title()} plot not found: {plot_file}")


def write_training_log(
    run_name,
    train_loss_history,
    train_acc_history,
    valid_loss_history,
    valid_acc_history,
    test_accuracy,
    best_validation_accuracy,
    config,
):
    """
    Write training results to a log file.

    :param run_name: Name of the training run
    :param train_loss_history: Training loss history
    :param train_acc_history: Training accuracy history
    :param valid_loss_history: Validation loss history
    :param valid_acc_history: Validation accuracy history
    :param test_accuracy: Final test accuracy
    :param best_validation_accuracy: Best validation accuracy achieved
    :param config: Training configuration dictionary
    """
    import json
    from datetime import datetime

    os.makedirs("plots", exist_ok=True)
    log_file = f"plots/{run_name}.log"

    with open(log_file, "w") as f:
        f.write(f"Training Log for: {run_name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Configuration: {config}\n\n")

        # Write epoch-by-epoch results
        f.write("Epoch,Train_Loss,Train_Acc,Valid_Loss,Valid_Acc\n")
        for epoch, (tl, ta, vl, va) in enumerate(
            zip(
                train_loss_history,
                train_acc_history,
                valid_loss_history,
                valid_acc_history,
            )
        ):
            f.write(f"{epoch},{tl:.6f},{ta:.6f},{vl:.6f},{va:.6f}\n")

        f.write(f"\nFinal Test Accuracy: {test_accuracy:.6f}\n")
        f.write(f"Best Validation Accuracy: {best_validation_accuracy:.6f}\n")

        # Write structured results as JSON for easy parsing
        results = {
            "train_loss_history": train_loss_history,
            "train_acc_history": train_acc_history,
            "valid_loss_history": valid_loss_history,
            "valid_acc_history": valid_acc_history,
            "test_accuracy": test_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "config": config,
            "run_name": run_name,
        }
        f.write(f"RESULTS:{json.dumps(results)}\n")

    print(f"Training log saved to: {log_file}")
