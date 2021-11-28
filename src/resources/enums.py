
from enum import Enum


class ContractType(Enum):
    STK = 1
    FUT = 2

class BarSize(Enum):
    SEC_01 = "1 sec"
    SEC_05 = "5 secs"
    SEC_10 = "10 secs"
    SEC_15 = "15 secs"
    SEC_30 = "30 secs"
    MIN_01 = "1 min"
    MIN_02 = "2 mins"
    MIN_03 = "3 mins"
    MIN_05 = "5 mins"
    MIN_10 = "10 mins"
    MIN_15 = "15 mins"
    MIN_20 = "20 mins"
    MIN_30 = "30 mins"
    HRS_01 = "1 hour"
    HRS_02 = "2 hours"
    HRS_03 = "3 hours"
    HRS_04 = "4 hours"
    HRS_08 = "8 hours"
    DAY_01 = "1 day"


# def BarDuration(barSize: BarSize, length: int) -> str:
#     return ""

BarDuration = {
    "1 sec": "1800 S",  # 30 mins
    "5 secs": "3600 S",  # 1 hr
    "10 secs": "14400 S",  # 4 hrs
    "15 secs": "14400 S",  # 4 hrs
    "30 secs": "28800 S",  # 8 hrs
    "1 min": "1 D",
    "2 mins": "2 D",
    "3 mins": "1 W",
    "5 mins": "1 W",
    "10 mins": "1 W",
    "15 mins": "1 W",
    "20 mins": "1 W",
    "30 mins": "1 M",
    "1 hour": "1 M",
    "2 hours": "1 M",
    "3 hours": "1 M",
    "4 hours": "1 M",
    "8 hours": "1 M",
    "1 day": "1 Y"
}