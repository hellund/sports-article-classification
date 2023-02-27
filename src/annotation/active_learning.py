import pandas as pd
from flask import Flask, request
from src.data.preprocessing import DataPreprocessorHelland
from sklearn.svm import SVC
import pickle
import os
from src.utils import get_project_root
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from src.annotation.doccano import get_latest_annotated_data
from skmultilearn.problem_transform import BinaryRelevance
import scipy
from skmultilearn.model_selection import iterative_train_test_split

app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict():
    """Flask api that retrieves an input from doccano and predicts the
    correct label using the largest active learning model and chosen
    preprocessing. The multi label binarizer is fetched and used to output
    labels as strings.

    Returns:
        dict: Dict with "label" as key and predicted label as value
    """
    new_input = request.get_json()
    print(f'Model: {find_newest_model()}')
    print('_' * 85 + '\n' + 'Text:\n' + f'{new_input["text"]}\n')

    loaded_model = pickle.load(open('active_learning_models\\' +
                                    find_newest_model(), 'rb'))
    loaded_mlb = pickle.load(open('active_learning_models\\' + 'mlb_' +
                                  find_newest_model(), 'rb'))
    preprocessor = DataPreprocessorHelland(pd.Series(new_input))
    preprocessor.preprocess()
    processed_input = preprocessor.text_series
    pred_labels = loaded_model.predict([processed_input['text']])
    pred_labels = loaded_mlb.inverse_transform(pred_labels)

    pred_proba = loaded_model.predict_proba([processed_input['text']])
    prob_dict = {k: v for k, v in
                 zip(loaded_mlb.classes_, pred_proba.toarray()[0])}

    print('Predicted labels:\n' + f'{pred_labels[0]}\n')
    print(f'Probabilities:')
    for key in prob_dict.keys():
        print(f'{key}: {prob_dict[key]}')

    print('\nMost likely:')
    for key in prob_dict.keys():
        if prob_dict[key] > 0.25:
            print(f'{key}: {prob_dict[key]}')
    print('\n')

    return {'label': pred_labels[0]}


def train_SVC_multilabel_model(X, y, C, gamma):
    """Trains a Binary Relevance SVC multi label model using inputted data
    and parameters and pickles the resulting model. The pickled model is
    saved in the active_learning_models directory with with a
    multilabelbinarizer object for mapping the model predictions back to
    strings. The inputted data is vectorized and inverse vectorized for the
    iterative_train_test_split from skmultilearn to work and to be compatible
    with new texts.

    Args:
        X (pd.Series): pd.Series with text data
        y (pd.Series): pd.Series with label data
        C (float): float with C value
        gamma (float): float with gamma value

    Returns:
        NoneType
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(y)
    y = mlb.transform(y)
    y = pd.DataFrame(y, columns=mlb.classes_)

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    X = vectorizer.inverse_transform(X_vectorized)
    X = pd.Series(X).apply(' '.join)

    y = scipy.sparse.csr_matrix(y.values)

    x_train, y_train, x_test, y_test = iterative_train_test_split(X_vectorized,
                                                                  y,
                                                                  test_size=0.3)

    x_train = vectorizer.inverse_transform(x_train)
    x_test = vectorizer.inverse_transform(x_test)

    x_train = pd.Series(x_train).apply(' '.join)
    x_test = pd.Series(x_test).apply(' '.join)

    model = BinaryRelevance(classifier=SVC(kernel='rbf', probability=True, C=C,
                                           gamma=gamma),
                            require_dense=[False, True])

    pipeline = make_pipeline(vectorizer, model)
    model = pipeline.fit(x_train, y_train)

    pred_labels_tr = model.predict(x_train)
    pred_labels_te = model.predict(x_test)


    print('----- Evaluation on Test Data -----')
    score_te = model.score(x_test, y_test)
    print('Accuracy Score: ', score_te)
    print(classification_report(y_test, pred_labels_te,
                                target_names=mlb.classes_))
    print('--------------------------------------------------------')

    print('----- Evaluation on Training Data -----')
    score_tr = model.score(x_train, y_train)
    print('Accuracy Score: ', score_tr)
    print(classification_report(y_train, pred_labels_tr,
                                target_names=mlb.classes_))
    print('--------------------------------------------------------')

    model = pipeline.fit(X, y)

    filename = f'active_learning_ml_model_{X.shape[0]}.pkl'
    pickle.dump(model, open('active_learning_models\\' + filename, 'wb'))
    pickle.dump(mlb, open('active_learning_models\\' + 'mlb_' + filename, 'wb'))


def find_newest_model():
    """Finds the largest model in the directory.

    Returns:
        str: String of the name of the larges/newest model
    """
    models = []
    root = get_project_root()
    for file in os.listdir(root + '/src/annotation/active_learning_models'):
        if file.endswith(".pkl"):
            models.append(file)
    splitted_models = [x.split('_') for x in models]
    model_with_number = [model[-1] for model in splitted_models]
    model_number = [model.split('.') for model in model_with_number]
    number = [int(number[0]) for number in model_number]
    return f'active_learning_ml_model_{max(number)}.pkl'


def run_train_ml_model():
    """Trains a SVC multi label model using the latest annotated data. All
    labels only occurring once are removed.

    Returns:
        NoneType
    """
    train = get_latest_annotated_data()
    preprocessor = DataPreprocessorHelland(train['text'])
    train['text'] = preprocessor.preprocess()
    exploded_train = train.explode('label')
    individual_label_count = exploded_train['label'].value_counts().to_dict()
    labels_with_enough_samples = [key for key in individual_label_count if
                                  individual_label_count[key] >= 2]

    trimmed_data = exploded_train[exploded_train['label'].isin(
        labels_with_enough_samples)]
    train = trimmed_data.groupby(['text'], as_index=False).agg(
        {'label': lambda x: x.tolist()})

    train_SVC_multilabel_model(train['text'], train['label'], C=2, gamma=0.55)


def main():
    """Trains a new SVC multi label model with the latest dataset from 
    doccano annotation and saves it to active_learning_models if __name__ == 
    "__main__".
    
    Returns:
        NoneType

    """
    run_train_ml_model()


if __name__ == '__main__':
    main()
