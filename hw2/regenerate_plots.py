"""Regenerate ALL plots from CSV data with larger fonts for the report."""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs")

# Global font settings — much larger for report readability
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 28,
    "lines.linewidth": 3,
    "lines.markersize": 8,
})


def read_csv(filepath):
    """Read CSV and return {column_name: [values]} with epoch list."""
    data = {}
    epochs = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"])
            for key, val in row.items():
                if key == "epoch":
                    continue
                if val == "":
                    continue
                if key not in data:
                    data[key] = []
                    epochs[key] = []
                data[key].append(float(val))
                epochs[key].append(ep)
    return data, epochs


def save_plot(epoch_list, value_list, filepath, title, ylabel, color="tab:blue"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epoch_list, value_list, color=color, linewidth=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  -> {filepath}")


def save_dual_plot(ep1, v1, label1, ep2, v2, label2, filepath, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ep1, v1, label=label1, linewidth=2.5)
    ax.plot(ep2, v2, label=label2, linewidth=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  -> {filepath}")


def regen_baseline():
    data, epochs = read_csv(os.path.join(LOGS, "baseline", "metrics.csv"))
    save_plot(epochs["ssl_loss"], data["ssl_loss"],
              os.path.join(LOGS, "baseline", "loss_curve.png"),
              "SimCLR Training Loss (Baseline)", "NT-Xent Loss", "tab:red")
    save_plot(epochs["knn_accuracy"], data["knn_accuracy"],
              os.path.join(LOGS, "baseline", "knn_curve.png"),
              "kNN Accuracy (Baseline, k=20)", "Top-1 Accuracy", "tab:blue")


def regen_ssl_probe():
    data, epochs = read_csv(os.path.join(LOGS, "ssl_probe", "metrics.csv"))
    save_plot(epochs["lp_test_acc"], data["lp_test_acc"],
              os.path.join(LOGS, "ssl_probe", "linear_probe_acc.png"),
              "Linear Probing Accuracy (SSL → CIFAR-10)", "Test Accuracy", "tab:green")


def regen_supervised():
    data, epochs = read_csv(os.path.join(LOGS, "supervised", "metrics.csv"))
    save_plot(epochs["sl_loss"], data["sl_loss"],
              os.path.join(LOGS, "supervised", "loss_curve.png"),
              "Supervised Training Loss", "Cross-Entropy Loss", "tab:red")
    save_dual_plot(
        epochs["sl_train_acc"], data["sl_train_acc"], "Train Accuracy",
        epochs["sl_test_acc"], data["sl_test_acc"], "Test Accuracy",
        os.path.join(LOGS, "supervised", "accuracy_curve.png"),
        "Supervised Accuracy", "Accuracy",
    )


def regen_temp01():
    data, epochs = read_csv(os.path.join(LOGS, "temp01", "metrics.csv"))
    save_plot(epochs["ssl_loss"], data["ssl_loss"],
              os.path.join(LOGS, "temp01", "loss_curve.png"),
              "SimCLR Loss (τ=0.1)", "NT-Xent Loss", "tab:red")
    save_plot(epochs["knn_accuracy"], data["knn_accuracy"],
              os.path.join(LOGS, "temp01", "knn_curve.png"),
              "kNN Accuracy (τ=0.1)", "Top-1 Accuracy", "tab:blue")


def regen_temp50():
    data, epochs = read_csv(os.path.join(LOGS, "temp50", "metrics.csv"))
    save_plot(epochs["ssl_loss"], data["ssl_loss"],
              os.path.join(LOGS, "temp50", "loss_curve.png"),
              "SimCLR Loss (τ=5.0)", "NT-Xent Loss", "tab:red")
    save_plot(epochs["knn_accuracy"], data["knn_accuracy"],
              os.path.join(LOGS, "temp50", "knn_curve.png"),
              "kNN Accuracy (τ=5.0)", "Top-1 Accuracy", "tab:blue")


def regen_no_proj():
    data, epochs = read_csv(os.path.join(LOGS, "no_proj", "metrics.csv"))
    save_plot(epochs["ssl_loss"], data["ssl_loss"],
              os.path.join(LOGS, "no_proj", "loss_curve.png"),
              "SimCLR Loss (No Projector)", "NT-Xent Loss", "tab:red")
    save_plot(epochs["knn_accuracy"], data["knn_accuracy"],
              os.path.join(LOGS, "no_proj", "knn_curve.png"),
              "kNN Accuracy (No Projector)", "Top-1 Accuracy", "tab:blue")


def regen_linear_probes():
    for name, title_suffix in [
        ("no_proj_probe", "No Projector → CIFAR-10"),
        ("temp01_probe", "τ=0.1 → CIFAR-10"),
        ("temp50_probe", "τ=5.0 → CIFAR-10"),
        ("random_probe", "Random → CIFAR-10"),
        ("ssl_cifar100", "SSL → CIFAR-100"),
        ("sl_cifar100", "Supervised → CIFAR-100"),
        ("random_cifar100", "Random → CIFAR-100"),
    ]:
        csv_path = os.path.join(LOGS, name, "metrics.csv")
        if not os.path.exists(csv_path):
            continue
        data, epochs = read_csv(csv_path)
        save_plot(epochs["lp_test_acc"], data["lp_test_acc"],
                  os.path.join(LOGS, name, "linear_probe_acc.png"),
                  f"Linear Probing ({title_suffix})", "Test Accuracy", "tab:green")


if __name__ == "__main__":
    print("Regenerating all plots with larger fonts...")
    regen_baseline()
    regen_ssl_probe()
    regen_supervised()
    regen_temp01()
    regen_temp50()
    regen_no_proj()
    regen_linear_probes()
    print("Done.")
