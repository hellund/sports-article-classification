import re
import pandas as pd
from src.data.nordskog_data import get_data
from src.utils import get_project_root
import nltk
from nltk.corpus import stopwords
from typing import List, Set

class DataPreprocessorNordskog:
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


class DataPreprocessorHelland:
    """A class to preprocess data from the football article scraping done by
    Helland.

    Args:
        text_series (pd.Series): Series with texts that is to be preprocessed

    Attributes:
        text_series (pd.DataFrame): DataFrame that is to be preprocessed
    """

    def __init__(self, text_series: pd.Series):
        self.text_series = text_series

    def remove_extra_spaces(self) -> pd.Series:
        """Removes unnecessary whitespaces from the text series attribute.

        Returns:
            pd.Series: Returns a copy of the text series attribute without
            extra spaces
        """
        self.text_series = self.text_series.map(
            lambda text: " ".join(str(text).split()))
        data = self.text_series.copy()
        return data

    def remove_non_ascii_characters(self):
        """Removes non ascii characters, but keeps "æøå", from the text
        series attribute.

        Returns:
            pd.Series: Returns a copy of the text series attribute with no
            ascii characters.
        """
        self.text_series = self.text_series.map(lambda text: re.sub('(['
                                                                    '^A-Za-zæøåüé–])+',
                                                                    ' ',
                                                                    str(text)))
        data = self.text_series.copy()
        return data

    def make_lower_cased(self):
        """Makes all characters lower cased in the text_series attribute.

        Returns:
            pd.Series: Returns a copy of the text series attribute with all
            lower cased text.
        """
        self.text_series = self.text_series.map(lambda text: str(text).lower())

    @staticmethod
    def remove_stopwords_from_string(text: List[str], stop_words: Set[str]):
        """Removes stopwords from a list of words.

        Args:
            text (List[str]): List of words that is going to be cleaned.
            stop_words (Set[str]): Set of words that are going to be removed.

        Returns:
            List[str]: List of words removed of stopwords.
        """
        filtered_text = []
        for word in text:
            if word not in stop_words:
                filtered_text.append(word)
        return filtered_text

    def remove_stopwords(self):
        """Initiates the removal of stopwords from class attribute text_series.

        Returns:
            pd.Series: Returns a copy of the text series attribute with all
            stopwords removed
        """
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        stop_words = set(stopwords.words('norwegian'))
        self.text_series = self.text_series.map(lambda x:
                                                self.remove_stopwords_from_string(
                                                    x.split(),
                                                    stop_words))
        self.text_series = self.text_series.map(lambda x: ' '.join(x))
        data = self.text_series.copy()
        return data

    def preprocess(self):
        """Applies all preprocessing actions on the text_series attribute.

        Returns:
            pd.Series: Returns a copy of the text series attribute with all
            preprocessing steps applied.
        """
        self.make_lower_cased()
        self.remove_non_ascii_characters()
        self.remove_extra_spaces()
        self.remove_stopwords()
        data = self.text_series.copy()
        return data


def make_csv_for_data_annotation():
    """Creates the original dataset for annotation from the scraped articles.

    Returns:
        NoneType
    """
    root = get_project_root()
    article_df = pd.read_csv(root + '/src/data/vg_articles_2022.csv',
                             encoding='utf-8-sig')
    paragraphs = article_df[article_df['tag'] == 'p'].copy()
    paragraphs = paragraphs.loc[:, ['text']].copy().reset_index(drop=True)
    paragraphs['labels'] = ''
    paragraphs.to_csv(root + '/src/data/annotation_dataset/vg_article_dataset'
                             '.csv',
                      encoding='utf-8-sig')


if __name__ == '__main__':
    make_csv_for_data_annotation()
