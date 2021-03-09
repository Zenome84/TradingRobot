
import os
import gc
import json
import glob
import math
import arrow
import random

from enum import Enum
from threading import Lock, Thread

from resources.ibapi_adapter import IBAPI
from resources.time_tools import wait_until
from models.asset import Asset


class RobotClient:
    class ContractType(Enum):
        STK = 1
        FUT = 2

    def __init__(self, connect=True):
        self.time_zone = 'US/Eastern'
        self.live = True
        self.clientId = 1

        if connect:
            self.connect_client()

        self.reqIds = set([0])
        self.reqIds_mutex = Lock()

        self.resolvedContracts = dict()
        self.contractDetailsObtained = dict()
        self.resolvedHistoricalTickData = dict()
        self.historicalTickDataObtained = dict()
        self.resolvedHistoricalBarData = dict()

        self.assetCache = dict()

        self.barTypesMinDuration = {
            '1 sec': '1800 S',  # 30 mins
            '5 sec': '3600 S',  # 1 hr
            '10 sec': '14400 S',  # 4 hrs
            '30 sec': '28800 S',  # 8 hrs
            '1 min': '1 D',
            '2 mins': '2 D',
            '3 mins': '1 W',
            '5 mins': '1 W',
            '10 mins': '1 W',
            '15 mins': '1 W',
            '20 mins': '1 W',
            '30 mins': '1 M',
            '1 hr': '1 M',
            '2 hrs': '1 M',
            '3 hrs': '1 M',
            '4 hrs': '1 M',
            '8 hrs': '1 M',
            '1 day': '1 Y'
        }

    def connect_client(self):
        self.tws_client = IBAPI(
            '127.0.0.1', 7496 if self.live else 7497, self.clientId, self)

        wait_until(
            condition_function=lambda: self.tws_client.isConnected,
            seconds_to_wait=5,
            msg="Waited more than 5 secs to establish connection"
        )

        # timePassed = 0
        # while not self.tws_client.isConnected():
        #     time.sleep(0.1)
        #     timePassed += 0.1
        #     if timePassed > 5:
        #         raise RuntimeError(
        #             "Waited more than 5 secs to establish connection")

    def disconnect_client(self):
        self.tws_client.disconnect()
        self.tws_client = None

    def client_connected(self):
        return self.tws_client is not None and self.tws_client.isConnected()

    def get_new_reqIds(self, n=1):
        if self.reqIds_mutex.acquire():
            next_reqId = max(self.reqIds) + 1
            new_reqIds = list(range(next_reqId, next_reqId + n))
            self.reqIds.add(*new_reqIds)
        self.reqIds_mutex.release()
        return new_reqIds

    def subscribe_contract(self, symbol, exchange, secType):
        asset_key = "{}@{}".format(symbol, exchange)
        self.assetCache[asset_key] = Asset(
            symbol=symbol, exchange=exchange, secType=secType, client=self)
        return asset_key


if __name__ == "__main__":

    robot_client = RobotClient()
    robot_client.subscribe_contract('ES', 'GLOBEX', 'FUT')
    robot_client.subscribe_contract('SPY', 'SMART', 'STK')
    robot_client.disconnect_client()
    print("Done")
