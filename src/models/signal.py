
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


class Signal:
    
    def __init__(self, length, name=None):
        '''
        mutex object that stores streaming data locally,
            keeps it up-to-date, and trims at length
        '''
        self.name = name
        self.length = length
        self.data = {}

    def updateData(self, )
