import pandas as pd
from flask import Flask, request
from src.data.preprocessing import DataPreprocessorHelland
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
from src.annotation.doccano import get_latest_annotated_data


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
    preprocessor = DataPreprocessorHelland(pd.Series(new_input))
    processed_input = preprocessor.text_series
    pred_labels = loaded_model.predict([processed_input['text']])
    pred_labels = loaded_mlb.inverse_transform(pred_labels)

    pred_proba = loaded_model.predict_proba([processed_input['text']])
    prob_dict = {k: v for k, v in zip(loaded_mlb.classes_, pred_proba[0])}

    print('Predicted labels:\n' + f'{pred_labels[0]}\n')
    print(f'Probabilities:\n{prob_dict}\n')

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
    mlb = MultiLabelBinarizer()
    mlb.fit(y)
    y = mlb.transform(y)
    y = pd.DataFrame(y, columns=mlb.classes_)
    vectorizer = TfidfVectorizer()
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
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
    run_train_ml_model()
    """Trains a new SVC multi label model with the latest dataset from 
    doccano annotation and saves it to active_learning_models if __name__ == 
    "__main__".
    
    Returns:
        NoneType

    """


# if __name__ == '__main__':
#     #main()

