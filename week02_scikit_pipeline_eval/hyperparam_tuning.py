import optuna
from logistic_regression_from_scratch import LogisticRegression
from pipeline import build_pipeline
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from data_loader import retrieve_data
import os
import json
import pickle

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log = True)
    lmbda = trial.suggest_float('lmbda', 1e-6, 1.0, log = True)
    reg = trial.suggest_categorical('reg', ['none', 'ridge', 'lasso'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    model_kwargs = dict(lr = lr, lmbda = lmbda, reg = reg,
                         batch_size = batch_size, verbose = False)
    trainval, _, _ = retrieve_data()
    X_trainval, y_trainval = trainval[:, :-1], trainval[:, -1]
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=611)
    aucs = []

    for train_idx, val_idx in kf.split(X_trainval, y_trainval):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
        model = build_pipeline(LogisticRegression(**model_kwargs))
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)
        aucs.append(roc_auc_score(y_val, probs))

    return np.mean(np.array(aucs).reshape(-1, ))

def save_best_model_and_params(study, X_trainval, y_trainval, save_path="models/best_model.pkl"):
    best_params = study.best_params
    model = build_pipeline(LogisticRegression(**best_params, verbose=False))
    model.fit(X_trainval, y_trainval)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

    with open("models/best_hyperparams.json", 'w') as f:
        json.dump(best_params, f, indent=4)

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    trainval, _, _ = retrieve_data()
    X_trainval, y_trainval = trainval[:, :-1], trainval[:, -1]
    save_best_model_and_params(study, X_trainval, y_trainval)

    print("Best Hyperparameters:")
    print(study.best_params)

if __name__ == '__main__':
    main()