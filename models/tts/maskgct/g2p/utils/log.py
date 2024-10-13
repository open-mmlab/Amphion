#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   log.py
@Time    :   2024/08/05 08:52:50
@Author  :   Bo Jin 
@Version :   1.0
@Contact :   jinbo5650@gmail.com
'''



import functools
import logging

__all__ = [
    'logger',
]


class Logger(object):
    def __init__(self, name: str=None):
        name = 'PaddleSpeech' if not name else name
        self.logger = logging.getLogger(name)

        log_config = {
            'DEBUG': 10,
            'INFO': 20,
            'TRAIN': 21,
            'EVAL': 22,
            'WARNING': 30,
            'ERROR': 40,
            'CRITICAL': 50,
            'EXCEPTION': 100,
        }
        for key, level in log_config.items():
            logging.addLevelName(level, key)
            if key == 'EXCEPTION':
                self.__dict__[key.lower()] = self.logger.exception
            else:
                self.__dict__[key.lower()] = functools.partial(self.__call__,
                                                               level)

        self.format = logging.Formatter(
            fmt='[%(asctime)-15s] [%(levelname)8s] - %(message)s')

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def __call__(self, log_level: str, msg: str):
        self.logger.log(log_level, msg)


logger = Logger()
