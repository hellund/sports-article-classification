import pandas as pd
import os
from typing import Tuple
import git


def download_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download training and testing data from Nordskogs repo and returning
    them as DataFrames.

    Returns:
        tuple[pd.Dataframe, pd.DataFrame]: Tuple containing the training and
        test data as DataFrames
    """
    train = pd.read_csv(
        'https://raw.githubusercontent.com/Halflingen/Text-Classification-Project/1ac2dbd8626e965ba60c29e8fb72fe5c53e0e951/Dataset/training_set.csv',
        names=['text', 'label']).reset_index(drop=True)
    test = pd.read_csv(
        'https://raw.githubusercontent.com/Halflingen/Text-Classification-Project/1ac2dbd8626e965ba60c29e8fb72fe5c53e0e951/Dataset/testing_set.csv',
        names=['text', 'label']).reset_index(drop=True)
    train.to_csv('training_data.csv', encoding='utf-8-sig')
    test.to_csv('testing_data.csv', encoding='utf-8-sig')
    return train, test


def get_git_root() -> str:
    """Finds the root of the current git repository that can be used to locate
    files independent on where the repository is cloned to.

    Returns:
        str: string containing root of git repository
    """
    path = os.getcwd()
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data from the current directory

    Returns:
        tuple[pd.Dataframe, pd.DataFrame]: Tuple containing the training and
        test data as DataFrames
    """
    root = get_git_root()
    train = pd.read_csv(root + '/src/data/training_data.csv',
                        names=['text', 'label'],
                        encoding='utf-8-sig').reset_index(drop=True)
    test = pd.read_csv(root + '/src/data/testing_data.csv',
                       names=['text', 'label'],
                       encoding='utf-8-sig').reset_index(drop=True)
    return train, test


def check_for_existing_data(root: str = None) -> bool:
    """Checks if the data already exist as csv files in the directory

    Args:
        root (str): string containing the root directory

    Returns:
        bool: boolean value that is True if both Nordskog's datafiles exist in
        directory
    """
    if root is None:
        root = get_git_root()
    if os.path.isfile(root + '/src/data/training_data.csv') \
            and os.path.isfile(root + '/src/data/testing_data.csv'):
        return True
    else:
        return False


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieves data from Nordskog from local directory or from his Github.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing data from
        Nordskog's master project
    """
    if not check_for_existing_data():
        train, test = download_data()
    else:
        train, test = load_data()
    return train, test


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function that makes sure that the data from Nordskog's thesis is
    downloaded to directory and returned as two dataframes

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing data from
        Nordskog's master project
    """
    train, test = get_data()
    return train, test


if __name__ == "__main__":
    training_data, test_data = main()
