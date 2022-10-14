
# import os
# import gc
# import json
# import glob
# import math
import time
import datetime
from ibapi.contract import Contract
from ibapi.order import Order
from typing import Dict, List
import arrow
# import random

# from enum import Enum
from threading import Lock, Thread

from resources.ibapi_adapter import IBAPI
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
        self.request_signal_map: Dict[int, str] = dict()
        self.order_asset_map: Dict[int, str] = dict()
        self.buffered_positions: Dict[str, int] = dict()

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

        # self.client_adapter.reqAccountUpdates(True, '')
        # self.client_adapter.reqPositions()
        self.nextValidOrderId = -1
        self.get_new_orderId()

    def client_connected(self):
        return self.client_adapter is not None and self.client_adapter.isConnected()

    def disconnect_client(self):
        for asset_key in list(self.assetCache.keys()):
            symbol, exchange = asset_key.split('@')
            self.unsubscribe_asset(symbol, exchange)
        self.client_adapter.cancelPositions()
        self.client_adapter.reqAccountUpdates(False, '')
        self.client_adapter.disconnect()
        self.client_adapter = None

    def get_new_orderId(self):
        self.orderIdObtained = False
        self.client_adapter.reqIds()
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

    def subscribe_asset(self, symbol, exchange, secType, num_agents = 0):
        asset_key = f"{symbol}@{exchange}"
        self.assetCache[asset_key] = Asset(
            symbol=symbol, exchange=exchange, secType=secType, client=self, num_agents=num_agents)
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
        for agentId in range(self.assetCache[asset_key].num_agents):
            for orderId in list(self.assetCache[asset_key].openOrders[agentId].keys()):
                self.cancelOrder(orderId)
        self.buffered_positions[asset_key] = self.assetCache[asset_key].position[0] # only important for IB with single account
        self.assetCache.pop(asset_key)
        return asset_key

    def subscribe_bar_signal(self, asset_key, bar_size, length):
        reqId = self.assetCache[asset_key].subscribe_bar_signal(
            bar_size, length)
        return reqId

    def unsubscribe_bar_signal(self, reqId):
        self.assetCache[self.request_signal_map[reqId]
                        ].unsubscribe_bar_signal(reqId)

    def placeOrder(self, asset_key, action, quantity, price, agentId = 0):
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
            self.assetCache[asset_key].order_agent_map[orderId] = agentId
            self.assetCache[asset_key].openOrders[agentId][orderId] = order
            self.assetCache[asset_key].openOrdersStatus[agentId][orderId] = 'PendingSubmit' # TODO: Enum state
            self.client_adapter.placeOrder(orderId, self.assetCache[asset_key].contract, order)
            wait_until(
                condition_function=lambda: self.assetCache[asset_key].openOrdersStatus[agentId].get(orderId, "Filled") in ['PreSubmitted','Submitted','Cancelled','Filled'],
                seconds_to_wait=5,
                msg=f"Waited more than 5 secs to submit order: {orderId}"
            )
        self.assetCache[asset_key].order_mutex.release()
        # TODO: check for success/fail
        return orderId

    def cancelOrder(self, orderId):
        asset_key = self.order_asset_map[orderId]
        agentId = self.assetCache[asset_key].order_agent_map[orderId]
        if self.assetCache[asset_key].order_mutex.acquire():
            self.assetCache[asset_key].openOrdersStatus[agentId][orderId] = 'PendingCancel' # TODO: Enum state
            self.client_adapter.cancelOrder(orderId)
            wait_until(
                condition_function=lambda: self.assetCache[asset_key].openOrdersStatus[agentId].get(orderId, "Filled") in ['Cancelled','Filled'],
                seconds_to_wait=5,
                msg=f"Waited more than 5 secs to cancel order: {orderId}"
            )
            self.assetCache[asset_key].cleanUpOrder(orderId, cancel=True)
        self.assetCache[asset_key].order_mutex.release()
        return orderId not in self.assetCache[asset_key].filledOrders[agentId]

    def updateOrder(self, orderId, quantity = None, price = None):
        if quantity is None and price is None:
            return # Do Nothing
        asset_key = self.order_asset_map[orderId]
        agentId = self.assetCache[asset_key].order_agent_map[orderId]
        action = self.assetCache[asset_key].openOrders[agentId][orderId].action
        if quantity is None:
            quantity = abs(self.assetCache[asset_key].openOrderQty[agentId][orderId])
        if price is None:
            price = self.assetCache[asset_key].openOrders[agentId][orderId].lmtPrice

        if self.cancelOrder(orderId):
            return self.placeOrder(asset_key, action, quantity, price, agentId)
        else:
            return orderId
    

    def updateBarData(self, reqId, bar):
        try:
            self.assetCache[self.request_signal_map[reqId]
                            ].updateBarData(reqId, bar)
        except:
            pass

    def updatePositionData(self, contract, position):
        asset_key = f"{contract.symbol}@{contract.exchange}"
        agentId = 0 # only support for IB adapter single agent
        if asset_key in self.assetCache:
            if self.assetCache[asset_key].position_mutex[agentId].acquire():
                if asset_key in self.assetCache:
                    self.assetCache[asset_key].position[agentId] = position
            self.assetCache[asset_key].position_mutex[agentId].release()
        else:
            self.buffered_positions[asset_key] = position

    def updateAccountData(self, key, val):
        self.account_info[key] = val
        # print(self.account_info)

    def handleOpenOrder(self, orderId, contract, order, orderState):
        asset_key = f"{contract.symbol}@{contract.exchange}"
        agentId = self.assetCache[asset_key].order_agent_map.get(orderId, None)
        if agentId is None or not (asset_key in self.assetCache and orderId in self.assetCache[asset_key].openOrders[agentId]):
            return
        if self.assetCache[asset_key].position_mutex[agentId].acquire():
            if orderId not in self.assetCache[asset_key].openOrders[agentId]:
                self.assetCache[asset_key].position_mutex[agentId].release()
                return
            self.assetCache[asset_key].openOrders[agentId][orderId] = order
            self.assetCache[asset_key].openOrdersStatus[agentId][orderId] = orderState.status
            if not self.assetCache[asset_key].updateOrderRulesObtained:
                self.assetCache[asset_key].initMargin = float(orderState.initMarginChange)
                self.assetCache[asset_key].maintMargin = float(orderState.maintMarginChange)
                self.assetCache[asset_key].commission = orderState.commission
                self.assetCache[asset_key].updateOrderRulesObtained = True
        self.assetCache[asset_key].position_mutex[agentId].release()
    
    def handleOrderStatus(self, orderId, status, filled, remaining, avgFillPrice):
        asset_key = self.order_asset_map.get(orderId, None)
        if asset_key is None:
            return
        agentId = self.assetCache[asset_key].order_agent_map[orderId]
        if self.assetCache[asset_key].position_mutex[agentId].acquire():
            if orderId not in self.assetCache[asset_key].openOrders[agentId]:
                self.assetCache[asset_key].position_mutex[agentId].release()
                return
            self.assetCache[asset_key].openOrdersStatus[agentId][orderId] = status
            self.assetCache[asset_key].openOrderQty[agentId][orderId] = remaining * self.assetCache[asset_key].get_order_sign(orderId)
            self.assetCache[asset_key].cleanUpOrder(orderId)
        self.assetCache[asset_key].position_mutex[agentId].release()

    def handleOrderExecution(self, execution):
        orderId = execution.orderId
        asset_key = self.order_asset_map.get(orderId, None)
        if asset_key is None:
            return
        agentId = self.assetCache[asset_key].order_agent_map[orderId]
        if self.assetCache[asset_key].position_mutex[agentId].acquire():
            if orderId not in self.assetCache[asset_key].openOrders[agentId]:
                self.assetCache[asset_key].position_mutex[agentId].release()
                return
            qty = execution.shares
            cumQty = execution.cumQty
            price = execution.avgPrice
            mult = self.assetCache[asset_key].get_order_sign(orderId)

            self.assetCache[asset_key].position[agentId] += qty*mult
            self.assetCache[asset_key].openOrderQty[agentId][orderId] = (self.assetCache[asset_key].openOrders[agentId][orderId].totalQuantity - cumQty)*mult
            self.assetCache[asset_key].cleanUpOrder(orderId)
            self.assetCache[asset_key].positionLogs[agentId].append((qty*mult, price))
        self.assetCache[asset_key].position_mutex[agentId].release()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # ClockController.set_utcnow(arrow.get(datetime.datetime(
    #     2020, 7, 15, 9, 30, 0), ClockController.time_zone))
    robot_client = RobotClient(cliendId=0, live=False)
    # robot_client = RobotClient(cliendId=0, simulator="influx")
    num_agents = 2
    es_key = robot_client.subscribe_asset('ES', 'GLOBEX', 'FUT', num_agents)
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
    
    # reqId5 = robot_client.subscribe_bar_signal(es_key, BarSize.SEC_01, 10000)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(4, 1, 1)
    # ax2 = fig.add_subplot(4, 1, 2)
    # ax3 = fig.add_subplot(4, 1, 3)
    # ax4 = fig.add_subplot(4, 1, 4)

    real_ts = arrow.utcnow()
    sim_ts = ClockController.utcnow()
    # eod_ts = arrow.get(datetime.datetime(2020, 7, 15, 16, 00, 0), ClockController.time_zone)
    eod_ts = ClockController.utcnow().replace(hour=16, minute=0, second=0)
    buyId  = dict()
    sellId = dict()
    logPnL: Dict[int, List[float]] = { n: [] for n in range(num_agents) }
    closePrice: Dict[int, List[float]] = { n: [] for n in range(num_agents) }
    stdPrice: Dict[int, List[float]] = { n: [] for n in range(num_agents) }

    while ClockController.utcnow() < eod_ts:
        data1 = robot_client.assetCache[es_key].signals[reqId1].get_numpy()
        data2 = robot_client.assetCache[es_key].signals[reqId2].get_numpy()
        data3 = robot_client.assetCache[es_key].signals[reqId3].get_numpy()
        data4 = robot_client.assetCache[es_key].signals[reqId4].get_numpy()
    
        # data5 = robot_client.assetCache[es_key].signals[reqId5].get_numpy()

        # print(f"Pending OrderIds:")
        # for orderId, order in robot_client.assetCache[es_key].openOrders[agentId].items():
        #     print(f"OrderId: {order.orderId} | Status: {robot_client.assetCache[es_key].openOrdersStatus[agentId][orderId]}")

        # high_price = (data1[-10:, 2] - data1[-10:, 7]).max()-0.25
        # low_price = (data1[-10:, 3] - data1[-10:, 7]).min()+0.25
        # weights = np.exp(-np.arange(9)/20)
        # trend = (np.diff(data1[-10:, 7], n=1) * weights).sum() / weights.sum() + data1[-1, 7]
        # high_price = round(4*(high_price + trend))/4
        # low_price = round(4*(low_price + trend))/4

        if ClockController.utcnow().int_timestamp % 50 == 0:
            for agentId in range(num_agents):
                print(
                    f"Agent: {agentId} || " +
                    f"Open [Long: {robot_client.assetCache[es_key].openLongQty(agentId):2d} | " +
                    f"Short: {robot_client.assetCache[es_key].openShortQty(agentId):2d}] | " +
                    f"Position: {robot_client.assetCache[es_key].position[agentId]:=+3d} | " +
                    f"RealizedPnL: {float(robot_client.assetCache[es_key].getPnL(agentId)):=+9.2f} | " +
                    # f"PnL: {float(robot_client.account_info['RealizedPnL']) + float(robot_client.account_info['UnrealizedPnL'])}"
                    f"Elapsed Time [Real: {(arrow.utcnow() - real_ts).seconds} s | Simulated: {(ClockController.utcnow() - sim_ts).seconds} s]"
                )
                logPnL[agentId] += [robot_client.assetCache[es_key].getPnL(agentId)]
                closePrice[agentId] += [data1[-1, 4]]
                stdPrice[agentId] += [data1[:, 4].std()]

        for agentId in range(num_agents):
            
            if agentId > 0:
                high_price = round(4*(data1[-5:, 2].mean() + (1 + agentId/num_agents)*data1[-5:, 2].std()))/4
                low_price = round(4*(data1[-5:, 3].mean() - (1 + agentId/num_agents)*data1[-5:, 3].std()))/4
            else:
                high_price = (data1[-10:, 2] - data1[-10:, 7]).max()-0.25
                low_price = (data1[-10:, 3] - data1[-10:, 7]).min()+0.25
                weights = np.exp(-np.arange(9)/20)
                trend = (np.diff(data1[-10:, 7], n=1) * weights).sum() / weights.sum() + data1[-1, 7]
                high_price = round(4*(high_price + trend))/4
                low_price = round(4*(low_price + trend))/4

            if robot_client.assetCache[es_key].openLongQty(agentId) == 0 and robot_client.assetCache[es_key].position[agentId] < 1:
                buyId[agentId] = robot_client.placeOrder(es_key, 'BUY', 1, low_price, agentId)
            elif robot_client.assetCache[es_key].openLongQty(agentId) > 0 and buyId[agentId] in robot_client.assetCache[es_key].openOrders[agentId] and abs(robot_client.assetCache[es_key].openOrders[agentId][buyId[agentId]].lmtPrice - low_price) > 0.25:
                buyId[agentId] = robot_client.updateOrder(buyId[agentId], price=low_price)

            if robot_client.assetCache[es_key].openShortQty(agentId) == 0 and robot_client.assetCache[es_key].position[agentId] > -1:
                sellId[agentId] = robot_client.placeOrder(es_key, 'SELL', 1, high_price, agentId)
            elif robot_client.assetCache[es_key].openShortQty(agentId) > 0 and sellId[agentId] in robot_client.assetCache[es_key].openOrders[agentId] and abs(robot_client.assetCache[es_key].openOrders[agentId][sellId[agentId]].lmtPrice - high_price) > 0.25:
                sellId[agentId] = robot_client.updateOrder(sellId[agentId], price=high_price)

        # ax1.clear()
        # ax2.clear()
        # ax3.clear()
        # ax4.clear()

        # ax1.plot(data1[:, 0], data1[:, 2:4])
        # ax1.plot(data1[:, 0], data1[:, 7])
        # ax1.plot(data1[-1, 0], data1[-1, 4], marker='o')

        # ax2.plot(data2[:, 0], data2[:, 2:4])
        # ax2.plot(data2[:, 0], data2[:, 7])
        # ax2.plot(data2[-1, 0], data2[-1, 4], marker='o')

        # ax3.plot(data3[:, 0], data3[:, 2:4])
        # ax3.plot(data3[:, 0], data3[:, 7])
        # ax3.plot(data3[-1, 0], data3[-1, 4], marker='o')

        # ax4.plot(data4[:, 0], data4[:, 2:4])
        # ax4.plot(data4[:, 0], data4[:, 7])
        # ax4.plot(data4[-1, 0], data4[-1, 4], marker='o')

        # ClockController.increment_utcnow(1)
        # time.sleep(0.025)
        time.sleep(1.)
        # plt.pause(0.05)
    # plt.show()

    robot_client.disconnect_client()
    time.sleep(1)
    for n in range(num_agents):
        allData = np.array([logPnL[n], closePrice[n], stdPrice[n]]).T
        allDataNorm = (allData - allData.mean(0))/allData.std(0)
        allDataCorr = allDataNorm.T @ allDataNorm / allDataNorm.shape[0]
        print(f"CorrMat for Agent {n}")
        print(allDataCorr)

    print("Done")
