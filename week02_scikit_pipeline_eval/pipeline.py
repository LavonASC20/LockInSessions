import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from logistic_regression_from_scratch import LogisticRegression
import pickle 
import seaborn as sns
import evaluation_suite
import pandas as pd
import os
import json

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

def run_pipeline_workflow(data = None, target_col = None, random_state = 611, save_file_path: str = 'models/LR_model.pkl'):
    # load and preprocess data
    if data is None or target_col is None:
        print('Incomplete data, showcasing EDA on default Titanic data')
        data = sns.load_dataset('titanic')
        target_col = 'survived'
    os.makedirs(os.path.dirname(save_file_path), exist_ok = True)

    print(f'Created save file path: {save_file_path}\n\n')

    print('Cleaning data...\n\n')
    data, feature_names = clean_data(data)
    feature_names = feature_names.tolist()
    X = data.drop(target_col, axis = 'columns').to_numpy()
    y = data[target_col].to_numpy()

    # train test split
    print('Train/test split...\n\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = random_state)

    # push data through pipeline using fit_pipeline()
    print('Constructing pipeline...\n\n')
    pipeline = build_pipeline(LogisticRegression())

    # train model
    print('Training model...\n\n')
    pipeline.fit(X_train, y_train)

    # predict / evaluation
    print('Predicting and gathering metrics...\n\n')
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    metrics = evaluation_suite.evaluate_classification(y_true = y_test, y_pred = preds, y_prob = probs)

    # plots / images
    evaluation_suite.plot_classification_results(y_true = y_test, y_pred = preds, probs = probs)
    print('Evaluation images saved in plots/eval_suite.png\n\n')

    # pickle / save model parameters
    print('Saving model parameters...\n\n')
    with open(save_file_path, 'wb') as file:
        pickle.dump(pipeline, file)
    
    print('Saving feature names...\n\n')
    with open('models/feature_names.txt', 'w') as file:
        json.dump(feature_names, file)

    print('Saving evaluation metrics...\n\n')
    with open('models/metrics.txt', 'w') as file:
        json.dump(metrics, file, indent = 4, default = convert_np)

    print('Pipeline workflow complete!\n\n')

def clean_data(data):
    sex_map = {'male': 0, 'female': 1}
    data['sex'] = data['sex'].map(sex_map)

    data = data.drop('embarked', axis = 'columns')

    pclass_OHE = pd.get_dummies(data['pclass'], drop_first = True).astype(int)
    data = pd.concat([data, pclass_OHE], axis = 1)
    data = data.drop('pclass', axis = 'columns')

    # 'embarked' seems to be the same column as 'embark_town'
    # embarked_OHE = pd.get_dummies(data['embarked'], drop_first = True).astype(int)
    # data = pd.concat([data, embarked_OHE], axis = 1)
    # data = data.drop('embarked', axis = 'columns')

    class_OHE = pd.get_dummies(data['class'], drop_first = True).astype(int)
    data = pd.concat([data, class_OHE], axis = 1)
    data = data.drop('class', axis = 'columns')

    who_OHE = pd.get_dummies(data['who'], drop_first = True).astype(int)
    data = pd.concat([data, who_OHE], axis = 1)
    data = data.drop('who', axis = 'columns')

    data['adult_male'] = data['adult_male'].astype(int)

    deck_OHE = pd.get_dummies(data['deck'], drop_first = True).astype(int)
    data = pd.concat([data, deck_OHE], axis = 1)
    data = data.drop('deck', axis = 'columns')

    embark_town_OHE = pd.get_dummies(data['embark_town'], drop_first = True).astype(int)
    data = pd.concat([data, embark_town_OHE], axis = 1)
    data = data.drop('embark_town', axis = 'columns')

    alive_map = {'no': 0, 'yes': 1}
    data['alive'] = data['alive'].map(alive_map)

    data['alone'] = data['alone'].astype(int)

    return data, data.columns


def main():
    run_pipeline_workflow(sns.load_dataset('titanic'), 'survived')
    print('Training complete, trained model in models/ , results in plots/')


if __name__ == '__main__':
    main()