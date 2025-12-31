import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_bicep():
    print("Evaluating Bicep Model...")
    # Load data
    test_df = pd.read_csv("core/bicep_model/test.csv")
    test_df.loc[test_df["label"] == "C", "label"] = 0
    test_df.loc[test_df["label"] == "L", "label"] = 1
    
    # Load scaler and model
    sc = load_model("core/bicep_model/model/input_scaler.pkl")
    model = load_model("core/bicep_model/model/LR_model.pkl")
    
    X = test_df.drop("label", axis=1)
    X = pd.DataFrame(sc.transform(X))
    y = test_df["label"].astype('int')
    
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    print(f"Bicep Accuracy: {acc:.4f}")
    plot_confusion_matrix(cm, ["Correct", "Loose Arm"], "Bicep Model Confusion Matrix", "bicep_confusion_matrix.png")
    return {"Exercise": "Bicep", "Accuracy": acc, "Confusion Matrix": cm.tolist()}

def evaluate_plank():
    print("Evaluating Plank Model...")
    # Load data
    test_df = pd.read_csv("core/plank_model/test.csv")
    test_df.loc[test_df["label"] == "C", "label"] = 0
    test_df.loc[test_df["label"] == "H", "label"] = 1
    test_df.loc[test_df["label"] == "L", "label"] = 2
    
    # Load scaler and model
    sc = load_model("core/plank_model/model/input_scaler.pkl")
    model = load_model("core/plank_model/model/LR_model.pkl")
    
    X = test_df.drop("label", axis=1)
    X = pd.DataFrame(sc.transform(X))
    y = test_df["label"].astype('int')
    
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    print(f"Plank Accuracy: {acc:.4f}")
    plot_confusion_matrix(cm, ["Correct", "High Back", "Low Back"], "Plank Model Confusion Matrix", "plank_confusion_matrix.png")
    return {"Exercise": "Plank", "Accuracy": acc, "Confusion Matrix": cm.tolist()}

def evaluate_squat():
    print("Evaluating Squat Model...")
    # Load data
    test_df = pd.read_csv("core/squat_model/test.csv")
    test_df.loc[test_df["label"] == "down", "label"] = 0
    test_df.loc[test_df["label"] == "up", "label"] = 1
    
    # Load scaler and model
    sc = load_model("core/squat_model/model/input_scaler.pkl")
    model = load_model("core/squat_model/model/squat_model.pkl")
    
    X = test_df.drop("label", axis=1)
    X = pd.DataFrame(sc.transform(X))
    y = test_df["label"].astype('int')
    
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    print(f"Squat Accuracy: {acc:.4f}")
    plot_confusion_matrix(cm, ["Down", "Up"], "Squat Model Confusion Matrix", "squat_confusion_matrix.png")
    return {"Exercise": "Squat", "Accuracy": acc, "Confusion Matrix": cm.tolist()}

if __name__ == "__main__":
    results = []
    results.append(evaluate_bicep())
    results.append(evaluate_plank())
    results.append(evaluate_squat())
    
    # Save metrics to text file
    with open("evaluation_metrics.txt", "w") as f:
        for res in results:
            f.write(f"Exercise: {res['Exercise']}\n")
            f.write(f"Accuracy: {res['Accuracy']:.4f}\n")
            f.write(f"Confusion Matrix:\n{np.array(res['Confusion Matrix'])}\n")
            f.write("-" * 20 + "\n")
            
    print("Visualizations generated: bicep_confusion_matrix.png, plank_confusion_matrix.png, squat_confusion_matrix.png")
    print("Metrics saved to evaluation_metrics.txt")
