import slack
import configparser


def sofus_alert():
    config = configparser.ConfigParser()
    config.read('my_config.ini')
    token = config['slack']['TOKEN']
    slack_client = slack.WebClient(token=token)
    slack_client.chat_postMessage(channel='#pythonalert', text='Script is '
                                                               'finished')
