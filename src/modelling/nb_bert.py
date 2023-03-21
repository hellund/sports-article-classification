from src.utils import get_project_root
import pandas as pd
from src.annotation.doccano import get_latest_annotated_data
from src.data.preprocessing import DataPreprocessorHelland
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import BertForSequenceClassification
from skorch.hf import HuggingfacePretrainedTokenizer
import torch
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
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


def preprocessor(data):
    data_preprocessor = DataPreprocessorHelland(data['text'])
    data_preprocessor.make_lower_cased()
    data_preprocessor.remove_extra_spaces()
    data['text'] = data_preprocessor.text_series
    one_hot = MultiLabelBinarizer()
    one_hot_label = one_hot.fit_transform(data['label'])
    one_hot_label = [list(map(float, x)) for x in one_hot_label]
    data['label'] = pd.Series(list(one_hot_label))
    return data


def create_dataset_dict(train_split, validation_split):
    train_dataset = Dataset.from_pandas(train_split)
    train_dataset = train_dataset.remove_columns(['__index_level_0__'])
    validation_dataset = Dataset.from_pandas(validation_split)
    validation_dataset = validation_dataset.remove_columns(['__index_level_0__'])
    split_dict = {'train': train_dataset, 'validation': validation_dataset}
    datasets = DatasetDict(split_dict)
    return datasets


class BertModule(nn.Module):
    def __init__(self, name, num_labels):
        super().__init__()
        self.name = name
        self.num_labels = num_labels

        self.reset_weights()

    def reset_weights(self):
        self.bert = BertForSequenceClassification.from_pretrained(
            self.name, num_labels=self.num_labels, problem_type="multi_label_classification"
        )

    def forward(self, **kwargs):
        predictions = self.bert(**kwargs)
        return predictions.logits


def create_nb_bert_pipeline():
    # Choose a tokenizer and BERT model that work together
    TOKENIZER = 'NbAiLab/nb-bert-large'
    PRETRAINED_MODEL = 'NbAiLab/nb-bert-large'

    # model hyper-parameters
    OPTMIZER = torch.optim.AdamW
    LR = 5e-5
    MAX_EPOCHS = 3
    CRITERION = nn.CrossEntropyLoss
    BATCH_SIZE = 16

    # device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline = Pipeline([
        ('tokenizer', HuggingfacePretrainedTokenizer(TOKENIZER)),
        ('net', NeuralNetClassifier(
            BertModule,
            train_split=None,
            module__name=PRETRAINED_MODEL,
            module__num_labels=22,
            optimizer=OPTMIZER,
            lr=LR,
            max_epochs=MAX_EPOCHS,
            criterion=CRITERION,
            batch_size=BATCH_SIZE,
            iterator_train__shuffle=True,
            #device=DEVICE,
        )),
    ])
    return pipeline


def main():
    data = get_training_data()
    data = preprocessor(data)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], random_state=100)
    y_train = MultiLabelBinarizer().fit_transform(y_train)
    # datasets = create_dataset_dict(train_split, validation_split)
    pipeline = create_nb_bert_pipeline()
    pipeline.fit(X_train, y_train)


if __name__ == '__main__':
    main()