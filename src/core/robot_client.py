
# import os
# import gc
# import json
# import glob
# import math
import time
import datetime
from ibapi.contract import Contract
from ibapi.order import Order
import pytz
from typing import Dict
import arrow
# import random

# from enum import Enum
from threading import Lock, Thread

from resources.ibapi_adapter import IBAPI
from resources.ibapi_orders import Orders
from resources.sim_adapter import INFLUX
from resources.time_tools import ClockController, wait_until
from resources.enums import BarDuration, BarSize
from models.asset import Asset


class RobotClient:

    def __init__(self, cliendId=0, live=True, connect=True, simulator=None):
        self.time_zone = "US/Eastern"
        self.live = live
        self.simulator = simulator
        self.clientId = cliendId

        self.account_info = dict()
        self.accountInfoKeys = [
            'TotalCashBalance',
            'RealizedPnL',
            'UnrealizedPnL',
            'LastUpdate'
        ]

        self.reqIds = set([0])
        self.reqIds_mutex = Lock()

        self.resolvedContracts = dict()
        self.contractDetailsObtained = dict()
        self.resolvedHistoricalTickData = dict()
        self.historicalTickDataObtained = dict()
        self.resolvedHistoricalBarData = dict()

        self.assetCache: Dict[str, Asset] = dict()
        self.request_signal_map = dict()
        self.order_asset_map = dict()
        self.buffered_positions = dict()

        if connect:
            self.connect_client()

    def connect_client(self):
        if self.simulator == "influx":
            self.client_adapter = INFLUX(
                "Zenome", "_W_lIVGTgVIfmloET33KC95vL9Qzx3hdIkePQTGNv5hlaGpLn-Oy1ndGN4LhEflBNoKSM1D3eddRXO_rY-FguA==", "http://localhost:8086", self)
        else:
            self.client_adapter = IBAPI(
                '127.0.0.1', 7496 if self.live else 7497, self.clientId, self)

        wait_until(
            condition_function=lambda: self.client_adapter.isConnected,
            seconds_to_wait=5,
            msg="Waited more than 5 secs to establish connection"
        )

        self.client_adapter.reqAccountUpdates(True, '')
        self.client_adapter.reqPositions()
        self.get_new_orderId()

    def client_connected(self):
        return self.client_adapter is not None and self.client_adapter.isConnected()

    def disconnect_client(self):
        for asset_key in list(self.assetCache.keys()):
            symbol, exchange = asset_key.split('@')
            self.unsubscribe_asset(symbol, exchange)
        self.client_adapter.reqAccountUpdates(False, '')
        self.client_adapter.cancelPositions()
        self.client_adapter.disconnect()
        self.client_adapter = None

    def get_new_orderId(self):
        self.orderIdObtained = False
        self.client_adapter.reqIds(-1)
        wait_until(
            condition_function=lambda: self.orderIdObtained,
            seconds_to_wait=5,
            msg="Waited more than 5 secs to get new nextValidOrderId"
        )
        return self.nextValidOrderId

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
        asset_key = f"{symbol}@{exchange}"
        self.assetCache[asset_key] = Asset(
            symbol=symbol, exchange=exchange, secType=secType, client=self)
        self.assetCache[asset_key].update_order_rules()
        if asset_key in self.buffered_positions:
            contract = Contract()
            contract.symbol = symbol
            contract.primaryExchange = exchange
            self.updatePositionData(contract, self.buffered_positions[asset_key])
        return asset_key

    def unsubscribe_asset(self, symbol, exchange):
        asset_key = f"{symbol}@{exchange}"
        for reqId in list(self.assetCache[asset_key].signals.keys()):
            self.unsubscribe_bar_signal(reqId)
        self.buffered_positions[asset_key] = self.assetCache[asset_key].position
        self.assetCache.pop(asset_key)
        return asset_key

    def subscribe_bar_signal(self, asset_key, bar_size, length):
        reqId = self.assetCache[asset_key].subscribe_bar_signal(
            bar_size, length)
        return reqId

    def unsubscribe_bar_signal(self, reqId):
        self.assetCache[self.request_signal_map[reqId]
                        ].unsubscribe_bar_signal(reqId)

    def placeOrder(self, asset_key, action, quantity, price):
        if self.assetCache[asset_key].order_mutex.acquire():
            orderId = self.get_new_orderId()
            #####
            order = Order()
            order.orderId = orderId
            order.action = action
            order.orderType = "LMT"
            order.totalQuantity = quantity
            order.lmtPrice = price
            order.transmit = True
            #####
            self.order_asset_map[orderId] = asset_key
            self.assetCache[asset_key].openOrders[orderId] = order
            self.assetCache[asset_key].openOrdersStatus[orderId] = 'PendingSubmit' # TODO: Enum state
            self.client_adapter.placeOrder(orderId, self.assetCache[asset_key].contract, order)
            wait_until(
                condition_function=lambda: self.assetCache[asset_key].openOrdersStatus.get(orderId, "Filled") in ['PreSubmitted','Submitted','Cancelled','Filled'],
                seconds_to_wait=5,
                msg=f"Waited more than 5 secs to submit order: {orderId}"
            )
        self.assetCache[asset_key].order_mutex.release()
        # TODO: check for success/fail
        return orderId

    def cancelOrder(self, orderId):
        asset_key = self.order_asset_map[orderId]
        if self.assetCache[asset_key].order_mutex.acquire():
            self.assetCache[asset_key].openOrdersStatus[orderId] = 'PendingCancel' # TODO: Enum state
            self.client_adapter.cancelOrder(orderId)
            wait_until(
                condition_function=lambda: self.assetCache[asset_key].openOrdersStatus.get(orderId, "Filled") in ['Cancelled','Filled'],
                seconds_to_wait=5,
                msg=f"Waited more than 5 secs to cancel order: {orderId}"
            )
            self.assetCache[asset_key].cleanUpOrder(orderId, cancel=True)
        self.assetCache[asset_key].order_mutex.release()
        # TODO: check for success/fail and return

    def updateOrder(self, orderId, quantity = None, price = None):
        if quantity is None and price is None:
            return # Do Nothing
        asset_key = self.order_asset_map[orderId]
        action = self.assetCache[asset_key].openOrders[orderId].action
        if quantity is None:
            quantity = abs(self.assetCache[asset_key].openOrderQty[orderId])
        if price is None:
            price = self.assetCache[asset_key].openOrders[orderId].lmtPrice
        self.cancelOrder(orderId)
        newOrderId = self.placeOrder(asset_key, action, quantity, price)
        # TODO: check for success/fail and return
        return newOrderId
    

    def updateBarData(self, reqId, bar):
        self.assetCache[self.request_signal_map[reqId]
                        ].updateBarData(reqId, bar)

    def updatePositionData(self, contract, position):
        asset_key = f"{contract.symbol}@{contract.primaryExchange}"
        if asset_key in self.assetCache:
            if self.assetCache[asset_key].position_mutex.acquire():
                if asset_key in self.assetCache:
                    self.assetCache[asset_key].position = position
            self.assetCache[asset_key].position_mutex.release()
        else:
            self.buffered_positions[asset_key] = position

    def updateAccountData(self, key, val):
        self.account_info[key] = val
        # print(self.account_info)

    def handleOpenOrder(self, orderId, contract, order, orderState):
        asset_key = f"{contract.symbol}@{contract.exchange}"
        if not (asset_key in self.assetCache and orderId in self.assetCache[asset_key].openOrders):
            return
        if self.assetCache[asset_key].position_mutex.acquire():
            if orderId not in self.assetCache[asset_key].openOrders:
                self.assetCache[asset_key].position_mutex.release()
                return
            self.assetCache[asset_key].openOrders[orderId] = order
            self.assetCache[asset_key].openOrdersStatus[orderId] = orderState.status
            if not self.assetCache[asset_key].updateOrderRulesObtained:
                self.assetCache[asset_key].initMargin = float(orderState.initMarginChange)
                self.assetCache[asset_key].maintMargin = float(orderState.maintMarginChange)
                self.assetCache[asset_key].commission = orderState.commission
                self.assetCache[asset_key].updateOrderRulesObtained = True
        self.assetCache[asset_key].position_mutex.release()
    
    def handleOrderStatus(self, orderId, status, filled, remaining, avgFillPrice):
        asset_key = self.order_asset_map.get(orderId, None)
        if asset_key is None:
            return
        if self.assetCache[asset_key].position_mutex.acquire():
            if orderId not in self.assetCache[asset_key].openOrders:
                self.assetCache[asset_key].position_mutex.release()
                return
            self.assetCache[asset_key].openOrdersStatus[orderId] = status
            self.assetCache[asset_key].openOrderQty[orderId] = remaining * self.assetCache[asset_key].get_order_sign(orderId)
            self.assetCache[asset_key].cleanUpOrder(orderId)
        self.assetCache[asset_key].position_mutex.release()

    def handleOrderExecution(self, execution):
        orderId = execution.orderId
        asset_key = self.order_asset_map.get(orderId, None)
        if asset_key is None:
            return
        if self.assetCache[asset_key].position_mutex.acquire():
            if orderId not in self.assetCache[asset_key].openOrders:
                self.assetCache[asset_key].position_mutex.release()
                return
            qty = execution.shares
            cumQty = execution.cumQty
            price = execution.avgPrice
            mult = self.assetCache[asset_key].get_order_sign(orderId)

            self.assetCache[asset_key].position += qty*mult
            self.assetCache[asset_key].openOrderQty[orderId] = self.assetCache[asset_key].openOrders[orderId].totalQuantity + cumQty*mult
            self.assetCache[asset_key].cleanUpOrder(orderId)
        self.assetCache[asset_key].position_mutex.release()

        # print(
        #     f"Execution: {execution} | "
        # )
        
# Execution: ExecId: 0000e1a7.61af68cc.01.01, Time: 20211207  16:01:53, Account: DU2870980,
# Exchange: GLOBEX, Side: SLD, Shares: 1.000000, Price: 4686.250000, PermId: 1879031808, ClientId: 0,
# OrderId: 117, Liquidation: 0, CumQty: 1.000000, AvgPrice: 4686.250000, OrderRef: , EvRule: , EvMultiplier: 0.000000, ModelCode: , LastLiquidity: 1 |

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # ClockController.set_utcnow(arrow.get(datetime.datetime(
    #     2020, 7, 13, 9, 30, 0), ClockController.time_zone))
    robot_client = RobotClient(cliendId=0, live=False)#, simulator="influx")
    es_key = robot_client.subscribe_asset('ES', 'GLOBEX', 'FUT')
    # es_key = robot_client.subscribe_asset('SPY', 'SMART', 'STK')

    # from resources.ibapi_orders import Orders
    # orders = Orders.BracketOrder(10, "BUY", 1, 4579.25, 4580.5, 4578.0)
    # for order in orders:
    #     robot_client.client_adapter.placeOrder(order.orderId, robot_client.assetCache[es_key].contract, order)
    
    wait_until(
        condition_function=lambda: robot_client.assetCache[es_key].updateOrderRulesObtained,
        seconds_to_wait=5,
        msg=f"Waited more than 5 secs to update order rules."
    )
    print(f"Asset: {es_key} | Commission: {robot_client.assetCache[es_key].commission} | Initial Margin: {robot_client.assetCache[es_key].initMargin} | Maintenance Marging: {robot_client.assetCache[es_key].maintMargin}")
    
    reqId1 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_01, 50)
    reqId2 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_05, 50)
    reqId3 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_30, 50)
    reqId4 = robot_client.subscribe_bar_signal(es_key, BarSize.HRS_01, 50)

    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    for i in range(15000):
        data1 = robot_client.assetCache[es_key].signals[reqId1].get_numpy()
        data2 = robot_client.assetCache[es_key].signals[reqId2].get_numpy()
        data3 = robot_client.assetCache[es_key].signals[reqId3].get_numpy()
        data4 = robot_client.assetCache[es_key].signals[reqId4].get_numpy()

        # print(f"Pending OrderIds:")
        # for orderId, order in robot_client.assetCache[es_key].openOrders.items():
        #     print(f"OrderId: {order.orderId} | Status: {robot_client.assetCache[es_key].openOrdersStatus[orderId]}")
        high_price = (data1[-10:, 2] - data1[-10:, 7]).max()-0.25
        low_price = (data1[-10:, 3] - data1[-10:, 7]).min()+0.25
        weights = np.exp(-np.arange(9)/20)
        trend = (np.diff(data1[-10:, 7], n=1) * weights).sum() / weights.sum() + data1[-1, 7]
        high_price = round(4*(high_price + trend))/4
        low_price = round(4*(low_price + trend))/4

        if i % 50 == 0:
            print(f"Open Long: {robot_client.assetCache[es_key].openLongQty} | Short: {robot_client.assetCache[es_key].openShortQty} | Position: {robot_client.assetCache[es_key].position}")

        # # 2: high; 3: low
        # high_price = round(4*(data1[-5:, 2].mean() + data1[-5:, 2].std()))/4
        # low_price = round(4*(data1[-5:, 3].mean() - data1[-5:, 3].std()))/4

        if robot_client.assetCache[es_key].openLongQty == 0 and robot_client.assetCache[es_key].position < 3:
            buyId = robot_client.placeOrder(es_key, 'BUY', 1, low_price)
        elif robot_client.assetCache[es_key].openLongQty > 0 and buyId in robot_client.assetCache[es_key].openOrders and abs(robot_client.assetCache[es_key].openOrders[buyId].lmtPrice - low_price) > 0.25:
            buyId = robot_client.updateOrder(buyId, price=low_price)

        if robot_client.assetCache[es_key].openShortQty == 0 and robot_client.assetCache[es_key].position > -3:
            sellId = robot_client.placeOrder(es_key, 'SELL', 1, high_price)
        elif robot_client.assetCache[es_key].openShortQty > 0 and sellId in robot_client.assetCache[es_key].openOrders and abs(robot_client.assetCache[es_key].openOrders[sellId].lmtPrice - high_price) > 0.25:
            sellId = robot_client.updateOrder(sellId, price=high_price)

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

        # ClockController.increment_utcnow(5)
        plt.pause(0.05)
    # plt.show()

    robot_client.disconnect_client()
    time.sleep(1)

    print("Done")
