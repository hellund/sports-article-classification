from flask import Flask, request
from src.data.active_learning_data import get_data
from src.data.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os
from src.utils import get_project_root
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict():
    """Flask api that retrieves an input from doccano and predicts the
    correct label using the largest active learning model.

    Returns:
        dict: Dict with "label" as key and predicted label as value
    """
    new_input = request.get_json()
    print('_' * 85 + '\n' + 'Text:\n' + f'{new_input["text"]}\n')

    loaded_model = pickle.load(open('active_learning_models\\' +
                                    find_newest_model(), 'rb'))
    loaded_mlb = pickle.load(open('active_learning_models\\' + 'mlb_' +
                                  find_newest_model(), 'rb'))

    pred_labels = loaded_model.predict([new_input['text']])
    pred_labels = loaded_mlb.inverse_transform(pred_labels)
    pred_proba = loaded_model.predict_proba([new_input['text']])
    prob_dict = {k: v for k, v in zip(loaded_mlb.classes_, pred_proba[0])}

    print('Predicted labels:\n' + f'{pred_labels[0]}\n')
    print(f'Probabilites:\n{prob_dict}\n')

    return {'label': pred_labels[0]}


def train_SVC_multilabel_model(X, y, C, gamma):
    """Trains a SVC multi label model using inputted data and parameters and
    pickles the resulting model. The pickled model is saved in the
    active_learning_models directory with with a multilabelbinarizer object
    for mapping the model predictions back to strings.

    Args:
        X (pd.Series): pd.Series with text data
        y (pd.Series): pd.Series with label data
        C (float): float with C value
        gamma (float): float with gamma value

    Returns:
        NoneType
    """
    y = y.map(lambda x: [x])

    mlb = MultiLabelBinarizer()
    mlb.fit(y)
    y = mlb.transform(y)
    vectorizer = TfidfVectorizer()

    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=0.1,
                                                        random_state=0)

    model = OneVsRestClassifier(SVC(kernel='rbf', probability=True, C=C,
                                    gamma=gamma))

    pipeline = make_pipeline(vectorizer, model)
    model = pipeline.fit(x_train, y_train)

    filename = f'active_learning_ml_model_{X.shape[0]}.pkl'
    pickle.dump(model, open('active_learning_models\\' + filename, 'wb'))
    pickle.dump(mlb, open('active_learning_models\\' + 'mlb_' + filename, 'wb'))
    # Predict class labels on training data
    pred_labels_tr = model.predict(x_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(x_test)

    print(mlb.inverse_transform(pred_labels_tr))

    # Use score method to get accuracy of the model
    print('----- Evaluation on Test Data -----')
    score_te = model.score(x_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')

    print('----- Evaluation on Training Data -----')
    score_tr = model.score(x_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')


def find_newest_model():
    """Finds the largest model in the directory.

    Returns:
        str: String of the name of the larges/newest model
    """
    models = []
    root = get_project_root()
    for file in os.listdir(root + '/src/modelling/active_learning_models'):
        if file.endswith(".pkl"):
            models.append(file)
    splitted_models = [x.split('_') for x in models]
    model_with_number = [model[-1] for model in splitted_models]
    model_number = [model.split('.') for model in model_with_number]
    number = [int(number[0]) for number in model_number]
    return f'active_learning_ml_model_{max(number)}.pkl'


def run_train_ml_model():
    """Trains a SVC multi label model using Nordskog data and preprocessing

    Returns:
        NoneType
    """
    train = get_data()

    preprocessor = DataPreprocessor(train)
    preprocessor.remove_paragraphs_over_65_words()
    preprocessor.remove_paragraphs_over_65_words()
    train = preprocessor.data.copy()
    train = train.groupby('label').filter(lambda x: len(x) >= 2)
    print(train['label'].value_counts())
    train_SVC_multilabel_model(train['text'], train['label'], C=2, gamma=0.55)


if __name__ == '__main__':
    run_train_ml_model()
