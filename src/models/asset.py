
from __future__ import annotations
from threading import Lock
from typing import Dict, TYPE_CHECKING

import time
import arrow

# from enum import Enum
# from threading import Lock, Thread
# from resources.ibapi_adapter import *
from ibapi.contract import Contract, ContractDetails
from ibapi.order import Order

from resources.ibapi_orders import Orders
if TYPE_CHECKING:
    from core.robot_client import RobotClient

from resources.enums import BarDuration, BarSize
from resources.time_tools import wait_until, ClockController
from models.signal import Signal


class Asset(Contract):

    def __init__(self, symbol: str, exchange: str, secType: str, client: RobotClient):
        '''
        a wrapper/collection to store all Signals related to one Contract
            secType is either 'FUT' or 'STK'
        '''
        if secType not in ['FUT', 'STK']:
            raise RuntimeError(
                f"secType must be 'FUT' or 'STK', but received: {secType}")

        self.client = client
        self.signals: Dict[str, Signal] = dict()
        self.order_mutex = Lock()
        self.openOrders = dict()
        self.openOrdersStatus = dict()
        self.position_mutex = Lock()
        self.openOrderQty = dict()
        self.position = 0
        self.initMargin = 0
        self.maintMargin = 0
        self.commission = 0

        self.contract = Contract()
        self.contract.symbol = symbol
        self.contract.exchange = exchange
        self.contract.currency = 'USD'
        self.contract.secType = secType

        if not self.client.client_connected():
            self.client.connect_client()

        reqId = self.client.get_new_reqIds()[0]
        self.client.contractDetailsObtained[reqId] = False
        self.client.client_adapter.reqContractDetails(
            reqId=reqId, contract=self.contract)

        wait_until(
            condition_function=lambda: self.client.contractDetailsObtained[reqId],
            seconds_to_wait=5,
            msg=f"Waited more than 5 secs to get contract details for: {symbol}@{exchange}"
        )

        self.client.contractDetailsObtained.pop(reqId)
        contracts = self.client.resolvedContracts.pop(reqId)
        self.client.resolve_request(reqId)
        
        self.updateOrderRulesObtained = False

        if secType == 'FUT':
            date_now = int(ClockController.utcnow().to(
                ClockController.time_zone).shift(days=-1).format("YYYYMMDD"))
            max_month = 30000000
            for contract in contracts:
                contract_month = int(contract.lastTradeDateOrContractMonth)
                if date_now < contract_month < max_month:
                    max_month = contract_month
                    self.contract = contract
        else:
            self.contract = contracts[0]

        self.asset_key = f"{symbol}@{exchange}"

    def cleanUpOrder(self, orderId, cancel = False):
        if not (self.openOrderQty.get(orderId, 1) == 0 or cancel):
            return
        if orderId in self.openOrderQty:
            self.openOrders.pop(orderId)
            self.openOrdersStatus.pop(orderId)
            self.openOrderQty.pop(orderId)
            self.client.order_asset_map.pop(orderId)

    def update_order_rules(self):
        if self.order_mutex.acquire():
            orderId = self.client.get_new_orderId()
            #####
            check_order_impact = Order()
            check_order_impact.orderId = orderId
            check_order_impact.action = "BUY"
            check_order_impact.orderType = "LMT"
            check_order_impact.totalQuantity = 1
            check_order_impact.lmtPrice = 1.
            check_order_impact.transmit = True
            check_order_impact.whatIf = True
            #####
            self.openOrders[orderId] = check_order_impact
            self.openOrdersStatus[orderId] = 'PendingSubmit' # TODO: Enum state
            self.updateOrderRulesObtained = False
            self.client.client_adapter.placeOrder(orderId, self.contract, check_order_impact)
            wait_until(
                condition_function=lambda: self.updateOrderRulesObtained,
                seconds_to_wait=5,
                msg=f"Waited more than 5 secs to update order rules: {orderId}"
            )
        self.order_mutex.release()
        self.openOrders.pop(orderId)
        self.openOrdersStatus.pop(orderId)

    def subscribe_bar_signal(self, bar_size: BarSize, length):
        if not self.client.client_connected():
            self.client.connect_client()

        reqId = self.client.get_new_reqIds()[0]
        self.signals[reqId] = Signal(
            length, f"{self.contract.symbol}@{self.contract.exchange}:{self.contract.exchange, bar_size.name}")
        self.client.resolvedHistoricalBarData[reqId] = False
        self.client.request_signal_map[reqId] = self.asset_key
        self.client.client_adapter.reqHistoricalData(
            reqId, self.contract, "", BarDuration[bar_size.value], bar_size.value, "TRADES", 0, 1, True, [])

        wait_until(
            condition_function=lambda: self.client.resolvedHistoricalBarData[reqId],
            seconds_to_wait=15,
            msg=f"Waited more than 15 secs to get {bar_size.value} bars for: {self.contract.symbol}@{self.contract.exchange}"
        )
        self.client.resolvedHistoricalBarData.pop(reqId)

        return reqId

    def unsubscribe_bar_signal(self, reqId):
        self.client.client_adapter.cancelHistoricalData(reqId)
        self.client.request_signal_map.pop(reqId)
        self.client.resolve_request(reqId)
        self.signals.pop(reqId)

    def updateBarData(self, reqId, bar):
        if reqId in self.signals:
            # if not (4000 <= bar.average <= 5000):
            #     print(
            #         f"Bad Average:: " +
            #         f"TS: {arrow.get(bar.date + ' ' + str(self.client.client_adapter.default_tz), 'YYYYMMDD  HH:mm:ss ZZZ')} | " +
            #         f"Open: {bar.open} | " +
            #         f"High: {bar.high} | " +
            #         f"Low: {bar.low} | " +
            #         f"Close: {bar.close} | " +
            #         f"Volume: {bar.volume} | " +
            #         f"Count: {bar.barCount} | " +
            #         f"VWAP: {bar.average}"
            #     )

            self.signals[reqId].updateData([
                arrow.get(bar.date + " " + ClockController.time_zone,
                        "YYYYMMDD  HH:mm:ss ZZZ").int_timestamp,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                0 if bar.volume < 0 else bar.volume,
                0 if bar.barCount < 0 else bar.barCount,
                self.signals[reqId].data[-1][4] if not (bar.low <= bar.average <= bar.high) else bar.average,
            ])

    def get_order_sign(self, orderId):
        if self.openOrders[orderId].action == 'SELL':
            return -1
        elif self.openOrders[orderId].action == 'BUY':
            return 1
        else:
            return 0

    @property
    def openShortQty(self):
        self.position_mutex.acquire()
        qty = -sum(qty for qty in self.openOrderQty.values() if qty < 0)
        self.position_mutex.release()
        return qty
        
    @property
    def openLongQty(self):
        self.position_mutex.acquire()
        qty = sum(qty for qty in self.openOrderQty.values() if qty > 0)
        self.position_mutex.release()
        return qty
