import logging
import os

import requests


def slack_notify(msg: str):
    token = os.environ.get('SLACK_API_TOKEN')
    if token is None:
        return
    try:
        requests.post(token, json={'text': msg})
    except:
        logging.warning('Unable to send notification to Slack!')


def print_and_slack(msg: str, *args, **kwargs):
    print(msg, *args, **kwargs)
    slack_notify(msg)


class SlackNotifier:
    def __init__(self, name: str):
        self.name = name
        self.msgs = [name]

    def write(self, msg: str):
        self.msgs.append(msg)
        return self

    def write_and_print(self, msg: str, *args, **kwargs):
        print(msg, *args, **kwargs)
        self.write(msg)
        return self

    def push(self):
        msg = '\n'.join(self.msgs)
        slack_notify(msg)
        self.msgs = [self.name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.push()
