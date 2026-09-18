#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2023 CESBIO / Centre National d'Etudes Spatiales
#
"""
MOdule for logging
"""

import logging
import typing as t
from typing import ClassVar

_T = t.TypeVar("_T")


class Singleton(type, t.Generic[_T]):
    """
    Singleton class
    """

    # _instances: dict[Singleton[_T], _T] = {}
    _instances = {}  # type: ignore #noqa: RUF012

    def __call__(cls, *args: t.Any, **kwargs: t.Any) -> _T:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class LoggerManager:
    """
    Class to manager logger through the modules
    """

    __metaclass__ = Singleton

    _loggers: ClassVar[dict[str, logging.Logger]] = {}

    _level = logging.INFO

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def get_logger(name=None):
        """
        Get logger
        """
        if not name:
            logging.basicConfig(
                level=LoggerManager._level,
                datefmt="%y-%m-%d %H:%M:%S",
                format="%(asctime)s :: %(levelname)s :: %(message)s",
            )
            return logging.getLogger()
        if name not in LoggerManager._loggers:
            logging.basicConfig(
                level=LoggerManager._level,
                datefmt="%y-%m-%d %H:%M:%S",
                format="%(asctime)s :: %(levelname)s :: %(message)s",
            )
            LoggerManager._loggers[name] = logging.getLogger(str(name))
        return LoggerManager._loggers[name]

    @staticmethod
    def set_level(level):
        """
        Set logger level
        """
        LoggerManager._level = level
        for name in LoggerManager._loggers:
            log = LoggerManager._loggers[name]
            log.setLevel(LoggerManager._level)
