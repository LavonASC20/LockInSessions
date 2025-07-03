import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def retrieve_data(seed = 611):
    data = sns.load_dataset('titanic')
    data, feature_names = clean_data(data)
    X = data.drop('survived', axis = 'columns').to_numpy()
    y = data['survived'].to_numpy()
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,
                                                               random_state=seed)
    trainval = np.hstack((X_trainval, y_trainval.reshape(-1, 1)))
    test = np.hstack((X_test, y_test.reshape(-1, 1)))

    return trainval, test, feature_names

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
