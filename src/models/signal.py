
# import os
# import gc
# import json
# import glob
# import math
# import arrow
# import random

import numpy as np

from collections import deque

# from enum import Enum
# from threading import Lock, Thread
from resources.ibapi_adapter import *


class Signal:

    def __init__(self, length, name=None):
        '''
        mutex object that stores streaming data locally,
            keeps it up-to-date, and trims at length
        '''
        self.name = name
        self.length = length
        self.data = deque(maxlen=length)

    def updateData(self, data_point):
        if len(self.data) > 0 and self.data[-1][0] == data_point[0]:
            self.data[-1] = data_point
        else:
            self.data.append(data_point)

    def get_numpy(self, N:int = None):
        if N is None:
            return np.array(self.data)
        else:
            return np.array(self.data)[-N:]
