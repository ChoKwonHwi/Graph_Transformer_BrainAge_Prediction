import pandas as pd
import matplotlib.pyplot as plt
# Assuming the data will be loaded from a CSV file named "loss_data.csv" with two columns: "train_loss" and "valid_loss"

# Load data from CSV

def loss_plot(loss_csv_path, loss_save_path) :
    df = pd.read_csv(loss_csv_path, names=["train_loss", "valid_loss"])
    df = df[(df["train_loss"] <= 1.8) & (df["valid_loss"] <= 1.8)]
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["train_loss"], label="Train Loss", color="blue")
    plt.plot(df.index, df["valid_loss"], label="Valid Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_save_path)
    plt.show()

def metric_plot(metric_csv_path, metric_save_path) :
    df = pd.read_csv(metric_csv_path, names=["Precision", "Sensitivity", "F1-score"])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Precision"], label="Precision", color="blue")
    plt.plot(df.index, df["Sensitivity"], label="Sensitivity", color="orange")
    plt.plot(df.index, df["F1-score"], label="F1-score", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Validation metrics per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(metric_save_path)
    plt.show()   

def precision_plot(metric_csv_path, precision_save_path) :
    df = pd.read_csv(metric_csv_path, names=["Precision", "Sensitivity", "F1-score"])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Precision"], label="Precision", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.title("Validation precision per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(precision_save_path)
    plt.show()   

def sensitivity_plot(metric_csv_path, sensitivity_save_path) :
    df = pd.read_csv(metric_csv_path, names=["Precision", "Sensitivity", "F1-score"])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Sensitivity"], label="Sensitivity", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Sensitivity")
    plt.title("Validation sensitivity per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(sensitivity_save_path)
    plt.show()   

def f1score_plot(metric_csv_path, f1score_save_path) :
    df = pd.read_csv(metric_csv_path, names=["Precision", "Sensitivity", "F1-score"])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["F1-score"], label="F1-score", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("F1-score")
    plt.title("Validation F1-score per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f1score_save_path)
    plt.show()   

loss_csv_path = "/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/loss_norm_MRI.csv"
loss_save_path = "/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/training_validation_loss_norm_plot.png"

loss_plot(loss_csv_path, loss_save_path)
#metric_plot(metric_csv_path, metric_save_path)
#precision_plot(metric_csv_path, precision_save_path)
#sensitivity_plot(metric_csv_path, sensitivity_save_path)
#f1score_plot(metric_csv_path, f1score_save_path)