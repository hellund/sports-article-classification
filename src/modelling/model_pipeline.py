import pandas as pd
import pickle
from src.annotation.doccano import get_latest_annotated_data
from src.utils import get_project_root
from sklearn.pipeline import Pipeline


def get_training_data():
    try:
        root = get_project_root()
        data = pd.read_pickle(root + '/src/data/training_data/multi_label_data.pkl')
    except FileNotFoundError as e:
        print(f"FileNotFoundError successfully handled\n"
              f"{e}")
        print('Pulling data from doccano instead:')
        print('"Remember to start doccano webserver and doccano task!"')
        data = get_latest_annotated_data().loc[:, ['text', 'label']].dropna().reset_index()
    return data


def create_ml_pipelines():
    pipelines = []
    pipelines.append(('NB_BERT' , Pipeline([
    ('tokenizer', HuggingfacePretrainedTokenizer(TOKENIZER)),
    ('net', NeuralNetClassifier(
        BertModule,
        module__name=PRETRAINED_MODEL,
        module__num_labels=len(set(y_train)),
        optimizer=OPTMIZER,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        criterion=CRITERION,
        batch_size=BATCH_SIZE,
        iterator_train__shuffle=True,
        device=DEVICE,
        callbacks=[
            LRScheduler(LambdaLR, lr_lambda=lr_schedule, step_every='batch'),
            ProgressBar(),
        ],
    )),
])))

    return pipelines


def run_ml_pipelines(pipelines, results_path):
    model_name = []
    results = []
    for pipe, model in pipelines:
        kfold = KFold(n_splits=10, random_state=100)
        cross_val_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        model_name.append(pipe)
        results.append(cross_val_results)
        printout = '%s: %f (%f)' % (model_name, cross_val_results.mean(), cross_val_results.std())
        print(printout)
    with open(results_path + '_,models.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(results_path + '_results.pkl', 'wb') as f:
        pickle.dump(results, f)


def main():
    data = get_training_data()
    pipelines = create_ml_pipelines()

    root = get_project_root()
    run_ml_pipelines(pipelines, root + '/src/modelling/results/run_01')


if __name__ == '__main__':
    main()
