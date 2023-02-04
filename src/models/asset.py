
from __future__ import annotations
from msilib.schema import Error
from threading import Lock
from typing import Dict, TYPE_CHECKING, List, Set, Tuple

import time
import arrow

# from enum import Enum
# from threading import Lock, Thread
# from resources.ibapi_adapter import *
from ibapi.contract import Contract, ContractDetails
from ibapi.order import Order

if TYPE_CHECKING:
    from core.robot_client import RobotClient

from resources.enums import BarDuration, BarSize
from resources.time_tools import wait_until, ClockController
from models.signal import Signal


class Asset:

    def __init__(self, symbol: str, exchange: str, secType: str, client: RobotClient, num_agents: int = 1):
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

        self.contract = Contract()
        self.contract.symbol = symbol
        self.contract.primaryExchange = exchange
        self.contract.currency = 'USD'
        self.contract.secType = secType

        self.initMargin = 0
        self.maintMargin = 0
        self.commission = 0

        self.num_agents = num_agents
        self.order_agent_map: Dict[int, int] = dict()
        self.openOrders: Dict[int, Dict[int, Order]] = { n: dict() for n in range(num_agents) }
        self.openOrdersStatus: Dict[int, Dict[int, str]] = { n: dict() for n in range(num_agents) }
        self.position_mutex = { n: Lock() for n in range(num_agents) }
        self.openOrderQty: Dict[int, Dict[int, int]] = { n: dict() for n in range(num_agents) }
        self.filledOrders: Dict[int, Set[int]] = { n: set() for n in range(num_agents) }
        self.positionLogs: Dict[int, List[Tuple[int, float]]] = { n: list() for n in range(num_agents) }
        self.position = { n: 0 for n in range(num_agents) }

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
        agentId = self.order_agent_map.get(orderId, None)
        if agentId is None or not (self.openOrderQty[agentId].get(orderId, 1) == 0 or cancel):
            return
        if orderId in self.openOrderQty[agentId]:
            if not cancel:
                self.filledOrders[agentId].add(orderId)
            self.openOrders[agentId].pop(orderId)
            self.openOrdersStatus[agentId].pop(orderId)
            self.openOrderQty[agentId].pop(orderId)
            self.order_agent_map.pop(orderId)
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
            self.openOrders[0][orderId] = check_order_impact
            self.openOrdersStatus[0][orderId] = 'PendingSubmit' # TODO: Enum state
            self.updateOrderRulesObtained = False
            self.order_agent_map[orderId] = 0
            self.client.client_adapter.placeOrder(orderId, self.contract, check_order_impact)
            wait_until(
                condition_function=lambda: self.updateOrderRulesObtained,
                seconds_to_wait=5,
                msg=f"Waited more than 5 secs to update order rules: {orderId}"
            )
        self.order_mutex.release()
        self.openOrders[0].pop(orderId)
        self.openOrdersStatus[0].pop(orderId)

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

        # try:
        wait_until(
            condition_function=lambda: self.client.resolvedHistoricalBarData[reqId],
            seconds_to_wait=15,
            msg=f"Waited more than 15 secs to get {bar_size.value} bars for: {self.contract.symbol}@{self.contract.primaryExchange}"
        )
        # except:
        #     self.unsubscribe_bar_signal(reqId)
        #     raise RuntimeError(f"Could not subscribe {bar_size.value} bars for: {self.contract.symbol}@{self.contract.primaryExchange}")
        # finally:
        self.client.resolvedHistoricalBarData.pop(reqId)

        return reqId

    def unsubscribe_bar_signal(self, reqId):
        self.client.client_adapter.cancelHistoricalData(reqId)
        self.client.request_signal_map.pop(reqId)
        self.client.resolve_request(reqId)
        self.signals.pop(reqId)

    def updateBarData(self, reqId, bar):
        if reqId in self.signals:
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
        agentId = self.order_agent_map[orderId]
        if self.openOrders[agentId][orderId].action == 'SELL':
            return -1
        elif self.openOrders[agentId][orderId].action == 'BUY':
            return 1
        else:
            return 0

    def openShortQty(self, agentId = 0):
        self.position_mutex[agentId].acquire()
        qty = -sum(qty for qty in self.openOrderQty[agentId].values() if qty < 0)
        self.position_mutex[agentId].release()
        return int(qty)

    def openLongQty(self, agentId = 0):
        self.position_mutex[agentId].acquire()
        qty = sum(qty for qty in self.openOrderQty[agentId].values() if qty > 0)
        self.position_mutex[agentId].release()
        return int(qty)

    def getPnL(self, agentId = 0):
        curr_price = self.getCurrPrice
        pnl = 0
        for pos, price in self.positionLogs[agentId].copy():
            pnl += (curr_price - price)*pos*50 - self.commission*abs(pos)
        return pnl

    @property
    def getCurrPrice(self):
        if len(self.signals) == 0:
            raise AttributeError(f"ERROR    Cannot get current price of {self.asset_key} without subscribing to at least 1 signal.")
        return self.signals[list(self.signals.keys())[0]].data[-1][4]
