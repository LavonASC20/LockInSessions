import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from logistic_regression_from_scratch import LogisticRegression
from data_loader import retrieve_data
import hyperparam_tuning
import pickle 
import seaborn as sns
import evaluation_suite
import pandas as pd
import os
import json
import argparse

def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def build_pipeline(model = None, random_state: int = 611):
    if model is None:
        model = LogisticRegression(random_state = random_state, batch_size = 64)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    return pipe

def run_pipeline_workflow(data = None, target_col = None, random_state = 611, save_file_path: str = 'models/best_model.pkl'):
    # load and preprocess data
    if data is None or target_col is None:
        print('Incomplete data, showcasing EDA on default Titanic data')
        trainval, test, feature_names = retrieve_data(seed = random_state)
    os.makedirs(os.path.dirname(save_file_path), exist_ok = True)

    print(f'Created save file path: {save_file_path}\n\n')

    # train test split
    print('Train/test split...\n\n')
    trainval, test, feature_names = retrieve_data(seed = random_state)
    X_train, y_train = trainval[:, :-1], trainval[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    # push data through pipeline using fit_pipeline()
    print('Constructing pipeline...\n\n')
    try:
        with open('models/best_model.pkl', 'rb') as f: # no training needed, load best model from tuning 
            pipeline = pickle.load(f)
    except FileNotFoundError: 
        raise RuntimeError('models/best_model.pkl not found. Try running hyperparam_tuning.py first.')
    
    # # train model
    # print('Training model...\n\n')
    # pipeline.fit(X_train, y_train)

    # predict / evaluation
    print('Predicting and gathering metrics...\n\n')
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    metrics = evaluation_suite.evaluate_classification(y_true = y_test, y_pred = preds, y_prob = probs)

    # plots / images
    evaluation_suite.plot_classification_results(y_true = y_test, y_pred = preds, probs = probs)
    print('Evaluation images saved in plots/eval_suite.png\n\n')

    # # pickle / save model parameters
    # print('Saving model parameters...\n\n')
    # with open(save_file_path, 'wb') as file:
    #     pickle.dump(pipeline, file)
    
    print('Saving feature names...\n\n')
    with open('models/feature_names.txt', 'w') as file:
        json.dump(feature_names.to_list(), file)

    print('Saving evaluation metrics...\n\n')
    with open('models/metrics.txt', 'w') as file:
        json.dump(metrics, file, indent = 4, default = convert_np)

    print('Pipeline workflow complete!\n\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action = 'store_true', 
                        help = 'Run hyperparameter tuning using Gaussian Process Bayesian Optimization')
    parser.add_argument('--run', action = 'store_true', 
                        help = 'Run evaluation on best saved model')
    args = parser.parse_args()

    if not (args.tune or args.run):
        print("No action specified. Use --tune and/or --run.")

    if args.tune:
        hyperparam_tuning.main()
        print('Hyperparameter tuning complete, params in models/')
    
    if args.run:
        run_pipeline_workflow(sns.load_dataset('titanic'), 'survived')
        print('Training complete, trained model in models/ , results in plots/')

if __name__ == '__main__':
    main()