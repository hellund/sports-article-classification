from src.utils import get_project_root
from src.annotation.doccano import get_latest_annotated_data
import pandas as pd



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