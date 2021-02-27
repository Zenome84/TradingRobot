
import os
import gc
import json
import glob
import math
import arrow
import random
from enum import Enum
from threading import Lock, Thread
from resources.ibapi_adapter import *
from ibapi.contract import Contract, ContractDetails
from resources.time_tools import ClockController


class Asset(Contract):
    
    def __init__(self, , symbol, exchange, secType, client=None):
        '''
        a wrapper/collection to store all Signals related to one Contract
            secType is either 'FUT' or 'STK'
        '''
        self.contract = Contract()
        self.contract.symbol = symbol
        self.contract.exchange = exchange
        self.contract.currency = 'USD'
        self.contract.secType = secType
        
        arrow.utcnow().to(TimeZone)

        if secType == 'FUT':
        else:
            self.contract.secType = 'FUT'

        contract.secType = 'FUT'
        contract.currency = 'USD'
        self.name = name
        self.length = length
        self.data = {}
