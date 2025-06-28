import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
from scipy import integrate

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', 
                            y_prob: np.ndarray = None):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average = average, zero_division = 0)
    prec = precision_score(y_true, y_pred, average = average, zero_division = 0)
    f1 = f1_score(y_true, y_pred, average = average, zero_division = 0)
    confusion = confusion_matrix(y_true, y_pred)
    metrics = {"accuracy": acc,
               "recall": rec,
               "precision": prec,
               "f1_score": f1,
               "confusion_matrix": confusion}
    
    if y_prob is not None:
        metrics["y_prob"] = y_prob
    
    return metrics

def plot_classification_results(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, thresholds: list = None):
    fig, ax = plt.subplots(2,2, figsize = (10, 10))
    fig.suptitle('Classification Results', fontsize=18, y=0.98)

    # predicted probability histogram detailing confidence in predictions
    pos = probs[y_pred == 1]
    neg = probs[y_pred == 0]
    ax[0,0].hist(pos, bins = len(set(pos)), label = 'Positive label', color = 'red', alpha = 0.6)
    ax[0,0].hist(neg, bins = len(set(neg)), label = 'Negative label', color = 'blue', alpha = 0.6)
    ax[0,0].legend()
    ax[0,0].set_xlabel('Predicted Probability')
    ax[0,0].set_ylabel('Count')
    ax[0,0].set_title('Class-Separated Predicted Probability Histogram')

    # confusion matrix
    sns.heatmap(confusion_matrix(y_true, y_pred), annot = True, fmt = 'd', cmap = 'Blues',
                          xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'], ax = ax[0,1])
    ax[0,1].set_title('Confusion Matrix')

    # ROC curve
    roc = compute_ROC(y_true, probs, thresholds)
    roc = roc[np.argsort(roc[:, 0])]
    auc = integrate.simpson(roc[:, 1], roc[:, 0])
    ax[1,1].plot(roc[:, 0], roc[:, 1], label = f'AUC: {auc:.3f}')
    ax[1,1].set_xlabel('False Positive Rate')
    ax[1,1].set_ylabel('True Positive Rate')
    ax[1,1].set_title('ROC Curve')
    ax[1,1].plot([0,1], [0,1], linestyle = 'dashed', color = 'purple')
    ax[1,1].legend(loc = 'lower right')
    ax[1,1].grid(True, linestyle='--', alpha=0.3)
    
    # PR curve
    pr = compute_PR(y_true, probs, thresholds)
    ax[1,0].plot(pr[:, 0], pr[:, 1])
    ax[1,0].set_xlabel('Recall')
    ax[1,0].set_ylabel('Precision')
    ax[1,0].set_title('PR Curve')
    ax[1,0].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    plt.savefig('plots/eval_suite.png')

def compute_ROC(y_true: np.ndarray, probs: np.ndarray, thresholds: list):
    if thresholds is None or any(np.array(thresholds)**2 > 1): # check for if thresholds are all between 0 and 1
        thresholds = np.linspace(0, 1, 100)
    
    res = np.zeros((len(thresholds), 2))
    for idx, t in enumerate(thresholds):
        y_pred = (probs >= t).astype(int)
        fp = np.sum(((y_pred == 1) & (y_true == 0)))
        tn = np.sum(((y_pred == 0) & (y_true == 0)))
        fpr = fp / (fp + tn)
        tpr = recall_score(y_true, y_pred, average = 'binary', zero_division=0)
        res[idx] = [fpr, tpr] # true positive rate is equivalent to recall
    return res.reshape(-1, 2)

def compute_PR(y_true: np.ndarray, probs: np.ndarray, thresholds: list):
    if thresholds is None or any(np.array(thresholds)**2 > 1): # check for if thresholds are all between 0 and 1
        thresholds = np.linspace(0, 1, 100)

    res = np.zeros((len(thresholds), 2))
    for idx, t in enumerate(thresholds):
        y_pred = (probs >= t).astype(int)
        res[idx] = [recall_score(y_true, y_pred, average = 'binary', zero_division=0), 
                    precision_score(y_true, y_pred, average = 'binary', zero_division=0)]
    return res.reshape(-1, 2)
