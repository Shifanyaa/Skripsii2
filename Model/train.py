import torch
import pandas as pd
import numpy as np
import argparse
import logging
import time
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, mean_squared_error
)
from models.dbn import DBN
from models.cnn import CNNModel
from models.random_forest import train_rf, evaluate_rf


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
    )


def load_data(p):
    df = pd.read_csv(p).dropna(subset=['Label'])
    y = np.where(df['Label'] > 0, 1, 0)
    X = df.drop(columns=['Label'])
    return X.values, y


def evaluate_torch(model, X, y):
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).squeeze()
        predb = (preds > 0.5).float().cpu().numpy()
        mcc = matthews_corrcoef(y, predb)
        auc = roc_auc_score(y, preds.cpu().numpy())
    return mcc, auc


def plot_metrics(model):
    window = 3
    mv = lambda lst: np.convolve(lst, np.ones(window) / window, mode='valid')
    epochs = range(1, len(model.train_losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, model.train_accs, label='Train Acc')
    plt.plot(epochs, model.val_accs, label='Val Acc', linestyle='--')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, model.train_losses, label='Train Loss')
    plt.plot(epochs, model.val_losses, label='Val Loss', linestyle='--', alpha=0.3)
    smoothed = mv(model.val_losses)
    plt.plot(range(window, len(model.val_losses) + 1), smoothed, label='Val Loss (MA)', linestyle='-')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "dbn", "rf"], required=True)
    args = parser.parse_args()

    setup_logging()
    X_train, y_train = load_data("train.csv")
    X_val, y_val = load_data("val.csv")
    X_test, y_test = load_data("test.csv")

    total_start = time.time()

    if args.model == "dbn":
        logging.info("Training DBN...")
        model = DBN(n_visible=X_train.shape[1])
        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_v = scaler.transform(X_val)
        X_ts = scaler.transform(X_test)

        Xtr = torch.tensor(X_tr, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32)
        Xv = torch.tensor(X_v, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.float32)

        model.train_model(Xtr, ytr, val_data=(Xv, yv), epochs=100, batch_size=256, lr=1e-4)
        plot_metrics(model)

        torch.save(model.state_dict(), "dbn_final.pth")
        joblib.dump(scaler, "scaler.pkl")
        logging.info("Model dan scaler tersimpan (dbn_final.pth, scaler.pkl)")

        mcc_val, auc_val = evaluate_torch(model, X_v, y_val)
        mcc_test, auc_test = evaluate_torch(model, X_ts, y_test)

        preds = model(torch.tensor(X_v, dtype=torch.float32)).squeeze().detach().cpu().numpy()
        fpr, tpr, _ = roc_curve(y_val, preds)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Validation)')
        plt.legend()
        plt.savefig("dbn_roc.png")
        plt.show()

        preds_bin = (preds > 0.5).astype(int)
        cm = confusion_matrix(y_val, preds_bin)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Validation)')
        plt.savefig("dbn_confusion_matrix.png")
        plt.show()

        precision, recall, _ = precision_recall_curve(y_val, preds)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Validation)')
        plt.savefig("dbn_precision_recall.png")
        plt.show()

        mse_val = mean_squared_error(y_val, preds)
        logging.info(f"Validation MSE: {mse_val:.6f}")

    elif args.model == "cnn":
        logging.info("Training CNN...")
        model = CNNModel(input_dim=X_train.shape[1])
        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_v = scaler.transform(X_val)
        X_ts = scaler.transform(X_test)

        Xtr = torch.tensor(X_tr, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32)
        Xv = torch.tensor(X_v, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.float32)

        model.train_model(Xtr, ytr, val_data=(Xv, yv), epochs=100, batch_size=128, lr=1e-4)
        plot_metrics(model)

        torch.save(model.state_dict(), "cnn_final.pth")
        joblib.dump(scaler, "scalercnn.pkl")
        logging.info("Model dan scaler tersimpan (cnn_final.pth, scalercnn.pkl)")

        mcc_val, auc_val = evaluate_torch(model, X_v, y_val)
        mcc_test, auc_test = evaluate_torch(model, X_ts, y_test)

        preds = model(torch.tensor(X_v, dtype=torch.float32)).squeeze().detach().cpu().numpy()
        fpr, tpr, _ = roc_curve(y_val, preds)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Validation)')
        plt.legend()
        plt.savefig("cnn_roc.png")
        plt.show()

        preds_bin = (preds > 0.5).astype(int)
        cm = confusion_matrix(y_val, preds_bin)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Validation)')
        plt.savefig("cnn_confusion_matrix.png")
        plt.show()

        precision, recall, _ = precision_recall_curve(y_val, preds)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Validation)')
        plt.savefig("cnn_precision_recall.png")
        plt.show()

        mse_val = mean_squared_error(y_val, preds)
        logging.info(f"Validation MSE: {mse_val:.6f}")

    elif args.model == "rf":
        logging.info("Training Random Forest...")
        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_v = scaler.transform(X_val)
        X_ts = scaler.transform(X_test)

        model = train_rf(X_tr, y_train)

        joblib.dump(model, "rf_final.pkl")
        joblib.dump(scaler, "scalerrf.pkl")
        logging.info("Model dan scaler tersimpan (rf_final.pkl, scalerrf.pkl)")

        mcc_val, auc_val = evaluate_rf(model, X_v, y_val)
        mcc_test, auc_test = evaluate_rf(model, X_ts, y_test)

        preds = model.predict_proba(X_v)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, preds)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Validation)')
        plt.legend()
        plt.savefig("rf_roc.png")
        plt.show()

        preds_bin = (preds > 0.5).astype(int)
        cm = confusion_matrix(y_val, preds_bin)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Validation)')
        plt.savefig("rf_confusion_matrix.png")
        plt.show()

        precision, recall, _ = precision_recall_curve(y_val, preds)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Validation)')
        plt.savefig("rf_precision_recall.png")
        plt.show()

        mse_val = mean_squared_error(y_val, preds)
        logging.info(f"Validation MSE: {mse_val:.6f}")

    total_end = time.time()
    logging.info(f"Total training time: {total_end - total_start:.2f}s")
    logging.info(f"Validation MCC: {mcc_val:.4f}, AUC: {auc_val:.4f}")
    logging.info(f"Test       MCC: {mcc_test:.4f}, AUC: {auc_test:.4f}")


if __name__ == "__main__":
    main()