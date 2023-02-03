import pandas as pd
from doccano_client import DoccanoClient
from src.utils import get_project_root
import zipfile
import os
from typing import Type
from pandas.errors import EmptyDataError


def login_to_doccano():
    """Instantiates a doccano client and logs in.

    Returns:
        Type[DoccanoClient]: DoccanoClient instance
    """
    # instantiate a client and log in to a Doccano instance
    client = DoccanoClient('http://127.0.0.1:8000')
    client.login(username='admin', password='password')
    return client


def download_data_from_doccano(client: Type[DoccanoClient]):
    """

    Args:
        client Type[DoccanoClient]: DoccanoClient instance

    Returns:
        NoneType
    """

    root = get_project_root()
    exported_file = client.data_export.download(project_id=3, format='CSV',
                                                dir_name=root + '/src/data/'
                                                                'annotation_'
                                                                'dataset',
                                                only_approved=True
                                                )

    with zipfile.ZipFile(exported_file, 'r') as zip_ref:
        zip_ref.extractall(root + '/src/data/annotation_dataset')

    os.replace(root + '/src/data/annotation_dataset/all.csv',
               root + '/src/data/annotation_dataset/'
                      'vg_article_annotated_dataset.csv')
    os.remove(exported_file)


def get_latest_annotated_data() -> pd.DataFrame:
    """Logs in to doccano and downloads the annotated dataset to
    annotation_dataset.

    Returns:
        pd.DataFrame: Dataframe containing the dataset from doccano
    """
    client = login_to_doccano()
    download_data_from_doccano(client)
    root = get_project_root()
    try:
        df = pd.read_csv(root +
                         '/src/data/annotation_dataset'
                         '/vg_article_annotated_dataset '
                         '.csv',
                         encoding='utf-8-sig')
    except EmptyDataError:
        print('No data! Check if data is approved.')
        raise EmptyDataError('No data! Check if data is approved.')
    df.drop(['id', 'Comments'], axis=1, inplace=True)
    df['label'] = df['label'].str.split('#')
    return df
