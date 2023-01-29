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

app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict():
    loaded_model = pickle.load(open(find_newest_model(), 'rb'))
    new_input = request.get_json()
    print(new_input)
    pred_labels = loaded_model.predict([new_input['text']])
    print(pred_labels[0])
    return {'label': str(pred_labels[0])}


def train_model(X, y, C, gamma):
    vectorizer = TfidfVectorizer()

    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)

    model = SVC(kernel='rbf', probability=True, C=C, gamma=gamma)

    pipeline = make_pipeline(vectorizer, model)
    model = pipeline.fit(x_train, y_train)

    filename = f'active_learning_model_{train.shape[0]}.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # Predict class labels on training data
    pred_labels_tr = model.predict(x_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(x_test)

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
    models = []
    root = get_project_root()
    for file in os.listdir(root + '/src/modelling'):
        if file.endswith(".pkl"):
            models.append(file)
    splitted_models = [x.split('_') for x in models]
    model_with_number = [model[-1] for model in splitted_models]
    model_number = [model.split('.') for model in model_with_number]
    number = [int(number[0]) for number in model_number]
    return f'active_learning_model_{max(number)}.pkl'


if __name__ == '__main__':
    train = get_data()
    preprocessor = DataPreprocessor(train)
    preprocessor.remove_paragraphs_over_65_words()
    preprocessor.remove_paragraphs_over_65_words()
    train = preprocessor.data.copy()
    train_model(train['text'], train['label'], C=2, gamma=0.55)

