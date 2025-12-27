"""Careless: Python implementation of the R careless package for detecting careless responding."""

from .evenodd import evenodd as evenodd
from .irv import irv as irv
from .longstring import longstring as longstring
from .mahad import mahad as mahad
from .psychsyn import psychant as psychant
from .psychsyn import psychsyn as psychsyn

__all__ = [
    "evenodd",
    "irv",
    "longstring",
    "mahad",
    "psychant",
    "psychsyn",
]
