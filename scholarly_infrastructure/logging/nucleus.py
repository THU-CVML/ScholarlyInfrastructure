"""我们结合loguru和rich的最佳实践，利用richuru库。"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../notebooks/02_logging (richuru).ipynb.

# %% auto 0
__all__ = ['rich_console', 'logger', 'original_print', 'print']

# %% ../../notebooks/02_logging (richuru).ipynb 3
# How to set logger level in loguru?
# https://github.com/Delgan/loguru/issues/138
# Make faster? picologging
# import 
# def set_logger_level(level):
#     os
# How to add file handler to loguru logger?
# try:
import richuru
from rich.console import Console
from rich.theme import Theme
import logging
from rich.markdown import Markdown
import rich

# 如果在python console里面调用，就可以看到好看的东西。
from rich import pretty
pretty.install()

rich_console = Console(
    theme=richuru.Theme(  # required, otherwise the color will be incorrect
        {
            'logging.level.success': 'green',
            'logging.level.trace': 'bright_black',
        }
    ), 
    markup=True
)
richuru.install(rich_console=rich_console, 
                time_format="%a %Y-%m-%d %H:%M:%S.%f", 
                level = logging.INFO
)
# except ImportError:
#     pass


# %% ../../notebooks/02_logging (richuru).ipynb 5
from loguru import logger

# class PropagateHandler(logging.Handler):
#     def emit(self, record: logging.LogRecord) -> None:
#         logging.getLogger(record.name).handle(record)

# logger.add(PropagateHandler(), format="{message}")
logger = logger

original_print = print
# print = lambda *args, **kwargs: logger.info(*args, **kwargs)
def print(*args, **kwargs):
    if len(args)==0 and len(kwargs) == 0:
        logger.info('')
    if len(args)==1 and len(kwargs) == 0:
        logger.info(args[0])
    else:
        logger.info(str(args), **kwargs)
