
# import os
# import gc
# import json
# import glob
# import math
import time
import arrow
# import random

# from enum import Enum
from threading import Lock, Thread

from resources.ibapi_adapter import IBAPI
from resources.time_tools import wait_until
from resources.enums import BarDuration, BarSize
from models.asset import Asset


class RobotClient:

    def __init__(self, cliendId=1, live=True, connect=True):
        self.time_zone = 'US/Eastern'
        self.live = live
        self.clientId = cliendId

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
        self.request_signal_map = dict()

    def connect_client(self):
        self.tws_client = IBAPI(
            '127.0.0.1', 7496 if self.live else 7497, self.clientId, self)

        wait_until(
            condition_function=lambda: self.tws_client.isConnected,
            seconds_to_wait=5,
            msg="Waited more than 5 secs to establish connection"
        )

    def client_connected(self):
        return self.tws_client is not None and self.tws_client.isConnected()

    def disconnect_client(self):
        self.tws_client.disconnect()
        self.tws_client = None

    def get_new_reqIds(self, n=1):
        if self.reqIds_mutex.acquire():
            next_reqId = max(self.reqIds) + 1
            new_reqIds = list(range(next_reqId, next_reqId + n))
            self.reqIds.add(*new_reqIds)
        self.reqIds_mutex.release()
        return new_reqIds

    def resolve_request(self, reqId):
        if self.reqIds_mutex.acquire():
            self.reqIds.remove(reqId)
        self.reqIds_mutex.release()

    def subscribe_asset(self, symbol, exchange, secType):
        asset_key = "{}@{}".format(symbol, exchange)
        self.assetCache[asset_key] = Asset(
            symbol=symbol, exchange=exchange, secType=secType, client=self)
        return asset_key

    def subscribe_bar_signal(self, asset_key, bar_size, length):
        reqId = self.assetCache[asset_key].subscribe_bar_signal(bar_size, length)
        return reqId

    def unsubscribe_bar_signal(self, reqId):
        self.assetCache[self.request_signal_map[reqId]].unsubscribe_bar_signal(reqId)

    def updateBarData(self, reqId, bar):
        self.assetCache[self.request_signal_map[reqId]].updateBarData(reqId, bar)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    robot_client = RobotClient(cliendId=2, live=True)
    es_key = robot_client.subscribe_asset('ES', 'GLOBEX', 'FUT')
    # es_key = robot_client.subscribe_asset('SPY', 'SMART', 'STK')
    reqId1 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_01, 50)
    reqId2 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_05, 50)
    reqId3 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_30, 50)
    reqId4 = robot_client.subscribe_bar_signal(es_key, BarSize.HRS_01, 50)

    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2)
    ax3 = fig.add_subplot(4,1,3)
    ax4 = fig.add_subplot(4,1,4)
    for i in range(1000):
        data1 = robot_client.assetCache[es_key].signals[reqId1].get_numpy()
        data2 = robot_client.assetCache[es_key].signals[reqId2].get_numpy()
        data3 = robot_client.assetCache[es_key].signals[reqId3].get_numpy()
        data4 = robot_client.assetCache[es_key].signals[reqId4].get_numpy()

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        ax1.plot(data1[:, 0], data1[:, 2:4])
        ax1.plot(data1[:, 0], data1[:, 7])
        ax1.plot(data1[-1, 0], data1[-1, 4], marker='o')
        
        ax2.plot(data2[:, 0], data2[:, 2:4])
        ax2.plot(data2[:, 0], data2[:, 7])
        ax2.plot(data2[-1, 0], data2[-1, 4], marker='o')
        
        ax3.plot(data3[:, 0], data3[:, 2:4])
        ax3.plot(data3[:, 0], data3[:, 7])
        ax3.plot(data3[-1, 0], data3[-1, 4], marker='o')
        
        ax4.plot(data4[:, 0], data4[:, 2:4])
        ax4.plot(data4[:, 0], data4[:, 7])
        ax4.plot(data4[-1, 0], data4[-1, 4], marker='o')

        plt.pause(0.05)
    # plt.show()

    robot_client.unsubscribe_bar_signal(reqId1)
    robot_client.unsubscribe_bar_signal(reqId2)
    robot_client.unsubscribe_bar_signal(reqId3)
    robot_client.unsubscribe_bar_signal(reqId4)

    robot_client.disconnect_client()
    print("Done")
