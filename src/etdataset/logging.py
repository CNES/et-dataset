#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales / Université Paul Sabatier (UT3)
#
"""
Logging module
"""

import logging


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances.keys():
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LoggerManager(object):
    __metaclass__ = Singleton

    _loggers = {}

    _level = logging.INFO

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def get_logger(name=None):
        if not name:
            logging.basicConfig(
                level=LoggerManager._level,
                datefmt="%y-%m-%d %H:%M:%S",
                format="%(asctime)s :: %(levelname)s :: %(message)s",
            )
            return logging.getLogger()
        elif name not in LoggerManager._loggers.keys():
            logging.basicConfig(
                level=LoggerManager._level,
                datefmt="%y-%m-%d %H:%M:%S",
                format="%(asctime)s :: %(levelname)s :: %(message)s",
            )
            LoggerManager._loggers[name] = logging.getLogger(str(name))
        return LoggerManager._loggers[name]

    @staticmethod
    def set_level(level):
        LoggerManager._level = level
        for name in LoggerManager._loggers.keys():
            log = LoggerManager._loggers[name]
            log.setLevel(LoggerManager._level)
