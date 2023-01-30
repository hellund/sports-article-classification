import pandas as pd
from src.utils import get_project_root


def get_data():
    """Retrieves data from directory ment for the active learning model.

    Returns:
        pd.Dataframe: Dataframe containing text and label
    """
    root = get_project_root()
    data = pd.read_csv(root + '/src/data/active_learning_VG_data.csv',
                       encoding='utf-8-sig')
    train = data.loc[:, ['text', 'label']]
    return train


if __name__ == '__main__':
    pass

