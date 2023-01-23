from setuptools import setup, find_packages

setup(
    name='Sports-article-classification packages',
    version='0.1.0',
    packages=find_packages(include=['src.data', 'src.data.*', 'src.slack_alert',
                                    'src.slack_alert.*'])
)
