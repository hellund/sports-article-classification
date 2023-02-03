import slack
import configparser
from src.utils import get_project_root


def sofus_alert() -> None:
    """Posts a message on the channel #pythonalert in my personal workspace

    Returns:
        NoneType
    """
    try:
        root = get_project_root()
        config = configparser.ConfigParser()
        config.read(root + '/src/slack_alert/my_config.ini')
        token = config['slack']['TOKEN']
        slack_client = slack.WebClient(token=token)
        slack_client.chat_postMessage(channel='#pythonalert', text='Script is '
                                                                   'finished')
        print('Sofus has sent an alert - Check slack!')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    sofus_alert()
