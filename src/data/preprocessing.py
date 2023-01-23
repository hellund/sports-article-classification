import re
import pandas as pd
from src.data.nordskog_data import get_data


class DataPreprocessor:
    """A class to preprocess data.

    Args:
        data (pd.DataFrame): DataFrame that is to be preprocessed

    Attributes:
        data (pd.DataFrame): DataFrame that is to be preprocessed
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def map_nordskog_data(self, numeric: bool = False) -> pd.DataFrame:
        """Function that maps Nordskog training data to correct labels with
        option of making the labels numeric.

        Args:
            numeric (bool): makes labels numeric if True

        Returns pd.DataFrame: DataFrame containing data with new mapped labels

        """
        old_target_labels = list(self.data['label'].value_counts().keys())
        new_target_labels = ['Goal/Assist', 'Quote', 'Transfer', 'Irrelevant',
                             'Ignore', 'Player Detail', 'Club Detail',
                             'Chances', 'Injuries', 'Rodt/Gult kort',
                             'Club Drama', 'Personal Drama']
        if numeric is True:
            new_target_labels = range(0, len(old_target_labels))
        label_mapping = {k: v for k, v in zip(old_target_labels,
                                              new_target_labels)}
        self.data['label'] = self.data['label'].map(label_mapping)
        data = self.data.copy()
        return data

    def limit_number_of_targets_to_5_and_merge(self, numeric: bool = False) -> \
            pd.DataFrame:
        """Function that limits the targets to only being the top 5 most
        frequent targets and merging the others to the most suited target.

        Args:
            numeric (bool): makes labels numeric if True

        Returns:
            pd.DataFrame: DataFrame containing only five targets
        """
        old_target_labels = list(self.data['label'].value_counts().keys())
        new_target_labels = ['Goal/Assist', 'Quote', 'Transfer', 'Irrelevant',
                             'Ignore', 'Ignore', 'Ignore', 'Goal/Assist',
                             'Irrelevant', 'Irrelevant', 'Irrelevant',
                             'Irrelevant']
        if numeric is True:
            new_target_labels = [0, 1, 2, 3, 4, 4, 4, 0, 3, 3, 3, 3]
        label_mapping = {k: v for k, v in zip(old_target_labels,
                                              new_target_labels)}
        self.data['label'] = self.data['label'].map(label_mapping)
        data = self.data.copy()
        return data

    def remove_extra_spaces_from_text(self) -> pd.DataFrame:
        """Function that removes extra spaces from data.

        Returns:
            pd.DataFrame: DataFrame without extra spaces
        """
        self.data['text'] = self.data['text'].map(lambda text:
                                                  re.sub(' +', ' ', text))
        data = self.data.copy()
        return data

    def remove_paragraphs_over_65_words(self) -> pd.DataFrame:
        """Function that removes paragraphs over 65 words from data.

        Returns:
            pd.DataFrame: DataFrame without texts over 65 words
        """
        self.data['num_words'] = self.data['text'].map(lambda x: len(x.split()))
        self.data = self.data[self.data['num_words'] < 65]
        self.data = self.data.drop(columns=['num_words'])
        data = self.data.copy()
        return data


def main() -> None:
    """Does some test using the DataPreprocessor class and prints out the
    results.

    Returns:
        NoneType
    """
    train, test = get_data()
    data_preprocessor = DataPreprocessor(train)
    mapped_data_categorical = data_preprocessor.map_nordskog_data()
    mapped_data_numerical = data_preprocessor.map_nordskog_data(numeric=True)
    mapped_to_5_categorical = \
        data_preprocessor.limit_number_of_targets_to_5_and_merge()
    mapped_to_5_numerical = \
        data_preprocessor.limit_number_of_targets_to_5_and_merge(numeric=True)

    more_spaceless_data = data_preprocessor.remove_extra_spaces_from_text()
    paragraph_limited_data = data_preprocessor.remove_paragraphs_over_65_words()

    print('Testing of DataPreprocessor \n' +'_'*50)
    print(mapped_data_categorical['label'].value_counts())
    print('_'*50)
    print(mapped_data_numerical['label'].value_counts())
    print('_'*50)
    print(mapped_to_5_categorical['label'].value_counts())
    print('_'*50)
    print(mapped_to_5_numerical['label'].value_counts())
    print('_'*50)
    print(more_spaceless_data['text'].head(5))
    print('_' * 50)
    print(f'Shape before 65 cleanse: {train.shape} --> Shape after: '
          f'{paragraph_limited_data.shape}')


if __name__ == '__main__':
    main()
