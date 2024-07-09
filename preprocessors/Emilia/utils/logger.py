# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import os


class Logger:
    """
    Logger class for managing logging operations.
    """

    _logger = None

    @classmethod
    def get_logger(cls, name=None):
        """
        Get the logger instance with the specified name. If it doesn't exist, create and cache it.

        Args:
            cls (type): The class type.
            name (str, optional): The name of the logger. Defaults to None, which uses the class name.

        Returns:
            logging.Logger: The logger instance.
        """
        if cls._logger is None:
            cls._logger = cls.init_logger(name)
        return cls._logger

    @classmethod
    def init_logger(cls, name=None):
        """
        Initialize the logger, including file and console logging.

        Args:
            cls (type): The class type.
            name (str, optional): The name of the logger. Defaults to None.

        Returns:
            logging.Logger: The initialized logger instance.
        """
        if name is None:
            name = "main"
            if "SELF_ID" in os.environ:
                name = name + "_ID" + os.environ["SELF_ID"]
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                name = name + "_GPU" + os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"Initialize logger for {name}")
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Add file handler to save logs to a file
        log_date = time.strftime("%Y-%m-%d", time.localtime())
        log_time = time.strftime("%H-%M-%S", time.localtime())
        os.makedirs(f"logs/{log_date}", exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(f"logs/{log_date}/{name}-{log_time}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Create a custom log formatter to set specific log levels to color
        class ColorFormatter(logging.Formatter):
            """
            Custom log formatter to add color to specific log levels.
            """

            def format(self, record):
                """
                Format the log record with color based on log level.

                Args:
                    record (logging.LogRecord): The log record to format.

                Returns:
                    str: The formatted log message.
                """
                if record.levelno >= logging.ERROR:
                    record.msg = "\033[1;31m" + str(record.msg) + "\033[0m"
                elif record.levelno >= logging.WARNING:
                    record.msg = "\033[1;33m" + str(record.msg) + "\033[0m"
                elif record.levelno >= logging.INFO:
                    record.msg = "\033[1;34m" + str(record.msg) + "\033[0m"
                elif record.levelno >= logging.DEBUG:
                    record.msg = "\033[1;32m" + str(record.msg) + "\033[0m"
                return super().format(record)

        color_formatter = ColorFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(color_formatter)
        logger.addHandler(ch)

        return logger


def time_logger(func):
    """
    Decorator to log the execution time of a function.

    Args:
        func (callable): The function whose execution time is to be logged.

    Returns:
        callable: The wrapper function that logs the execution time of the original function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        Logger.get_logger().debug(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute"
        )
        return result

    return wrapper
