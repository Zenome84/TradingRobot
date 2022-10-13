
from __future__ import annotations
from typing import Dict, TYPE_CHECKING
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.execution import Execution

import pytz
import time
import arrow
import copy
import queue
from decimal import Decimal

from threading import Lock, Thread

# from ibapi.wrapper import EWrapper
# from ibapi.client import EClient
from ibapi.utils import iswrapper
# from ibapi.order import Order
from ibapi.contract import Contract, ContractDetails
from ibapi.common import ListOfHistoricalTickLast, OrderId, TagValueList, BarData, TickerId
from ibapi.ticktype import TickTypeEnum

from influxdb_client import InfluxDBClient
# from influxdb_client.client.write_api import SYNCHRONOUS
if TYPE_CHECKING:
    from core.robot_client import RobotClient

from resources.time_tools import ClockController, wait_until


class ApiController:
    def __init__(self, msgHandler: RobotClient = None, time_zone=ClockController.time_zone):
        self.msgHandler = msgHandler
        self.default_tz = pytz.timezone(time_zone)
        self._acc_info = dict()

        if msgHandler is None:
            self.resolved_contract = []
            self.contractDetailsIsObtained = False

    @iswrapper
    def error(self, reqId, errorCode, errorString):
        if errorCode != 300:
            print("Error. Id: ", reqId, " Code: ",
                errorCode, " Msg: ", errorString)

    @iswrapper
    def connectAck(self):
        print("\n[Connected]")
        time.sleep(0.1)

    ###########################################
    
    # reqAccountUpdates
    @iswrapper
    def updateAccountValue(self, key: str, val: str, currency: str,
                           accountName: str):
        if self.msgHandler is not None:
            if key in self.msgHandler.accountInfoKeys and currency in ['USD']:
                self.msgHandler.updateAccountData(key, val)
        else:
            if key in ['TotalCashBalance','RealizedPnL','UnrealizedPnL'] and currency in ['USD']:
                print(
                    f"Account: {accountName} | " +
                    f"{key}: {val} {currency} | "
                )


    @iswrapper
    def updatePortfolio(self, contract: Contract, position: Decimal,
                        marketPrice: float, marketValue: float,
                        averageCost: float, unrealizedPNL: float,
                        realizedPNL: float, accountName: str):
        if self.msgHandler is not None:
            self.msgHandler.updatePositionData(contract, position)
        else:
            print(
                f"Account: {accountName} | " +
                f"Asset: {contract.symbol}@{contract.primaryExchange} | " +
                f"Position: {position} | " +
                f"MarketPrice: {marketPrice} | " +
                f"MarketValue: {marketValue} | " +
                f"AverageCost: {averageCost} | " +
                f"UnrealizedPNL: {unrealizedPNL} | " +
                f"RealizedPNL: {realizedPNL} | "
            )

    @iswrapper
    def updateAccountTime(self, timeStamp: str):
        if self.msgHandler is not None:
            if 'LastUpdate' in self.msgHandler.accountInfoKeys:
                self.msgHandler.updateAccountData('LastUpdate', timeStamp)
        else:
            print(
                f"TS: {timeStamp} | "
            )

    ###########################################

    # reqIds
    @iswrapper
    def nextValidId(self, orderId: int):
        if self.msgHandler is not None:
            self.msgHandler.nextValidOrderId = orderId
            self.msgHandler.orderIdObtained = True
        else:
            self.nextValidOrderId = orderId
            print(f"nextValidOrderId: {orderId}")

    ###########################################

    # placeOrder
    @iswrapper
    def openOrder(self, orderId: OrderId, contract: Contract, order: Order,
                  orderState: OrderState):
        if self.msgHandler is not None:
            self.msgHandler.handleOpenOrder(orderId, contract, order, orderState)
        else:
            print(
                f"OrderId: {orderId} | " +
                f"Asset: {contract.symbol}@{contract.exchange} | " +
                f"Action: {order.action} | " +
                f"OrderType: {order.orderType} | " +
                f"TotalQty: {order.totalQuantity} | " +
                f"LmtPrice: {order.lmtPrice} | " +
                f"Status: {orderState.status} | " +
                f"OrderState: {orderState.__dict__} | "
            )

    @iswrapper
    def orderStatus(self, orderId: OrderId, status: str, filled: Decimal,
                    remaining: Decimal, avgFillPrice: float, permId: int,
                    parentId: int, lastFillPrice: float, clientId: int,
                    whyHeld: str, mktCapPrice: float):
        if self.msgHandler is not None:
            self.msgHandler.handleOrderStatus(orderId, status, filled, remaining, avgFillPrice)
        else:
            print(
                f"OrderId: {orderId} | " +
                f"Filled: {filled} | " +
                f"Remaining: {remaining} | " +
                f"AvgFillPrice: {avgFillPrice} | " +
                f"WhyHeld: {whyHeld} | " +
                f"Status: {status} | " +
                f"MktCapPrice: {mktCapPrice} | "
            )

    @iswrapper
    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        if self.msgHandler is not None:
            self.msgHandler.handleOrderExecution(execution)
        else:
            print(
                f"ReqId: {reqId} | " +
                f"Asset: {contract.symbol}@{contract.exchange} | " +
                f"Execution: {execution} | "
            )

    ###########################################

    # reqContractDetails
    @iswrapper
    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        if self.msgHandler is not None:
            self.msgHandler.resolvedContracts.setdefault(
                reqId, []).append(contractDetails.contract)
        else:
            print(f"Contract Details: {contractDetails.contract}")
            self.resolved_contract.append(contractDetails.contract)

    @iswrapper
    def contractDetailsEnd(self, reqId: int):
        if self.msgHandler is not None:
            self.msgHandler.contractDetailsObtained[reqId] = True
        else:
            print(f"Completed Request: {reqId}")
            self.contractDetailsIsObtained = True

    ###########################################

    # reqHistoricalData, cancelHistoricalData for Update
    @iswrapper
    def historicalData(self, reqId: int, bar: BarData):
        if self.msgHandler is not None:
            self.msgHandler.updateBarData(reqId, bar)
        else:
            print(
                f"TS: {arrow.get(bar.date + ' ' + str(self.default_tz), 'YYYYMMDD  HH:mm:ss ZZZ')} | " +
                f"Open: {bar.open} | " +
                f"High: {bar.high} | " +
                f"Low: {bar.low} | " +
                f"Close: {bar.close} | " +
                f"Volume: {bar.volume} | " +
                f"Count: {bar.barCount} | " +
                f"VWAP: {bar.average}"
            )

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        if self.msgHandler is not None:
            self.msgHandler.resolvedHistoricalBarData[reqId] = True
        else:
            # super().historicalDataEnd(reqId, start, end)
            print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)

    @iswrapper
    def historicalDataUpdate(self, reqId: int, bar: BarData):
        self.historicalData(reqId, bar)

    ###########################################

    # reqHistoricalTicks for "TRADES" only
    @iswrapper
    def historicalTicksLast(self, reqId: int, ticks: ListOfHistoricalTickLast, done: bool):
        if self.msgHandler is not None:
            self.msgHandler.resolvedHistoricalTickData[reqId] = ticks
            self.msgHandler.historicalTickDataObtained[reqId] = True
        else:
            for tick in ticks:
                print(
                    f"HistoricalTickLast. Time: {arrow.get(tick.time).to(self.default_tz)} | " +
                    f"PastLimit: {tick.tickAttribLast.pastLimit} | " +
                    f"Unreported: {tick.tickAttribLast.unreported} | " +
                    f"Price: {tick.pric} | " +
                    f"Size: {tick.size} | " +
                    f"Exchange: {tick.exchange} | " +
                    f"SpecialConditions: {tick.specialConditions}"
                )
            print(f"Completed Request: {reqId} | Count: {len(ticks)}")

    ###########################################

    # @iswrapper
    # def tickPrice(self, reqId, tickType, price, attrib):
    #     # Do something with trading price
    #     print("Tick Price. Ticker Id: {} | tickType: {} | Price: {} | attr: {}".format(
    #         reqId, TickTypeEnum.to_str(tickType), price, attrib))

    # @iswrapper
    # def tickSize(self, reqId, tickType, size):
    #     # Do something with trading volume
    #     print("Tick Price. Ticker Id: {} | tickType: {} | Size: {}".format(
    #         reqId, TickTypeEnum.to_str(tickType), size))

    # @iswrapper
    # def realtimeBar(self, reqId: TickerId, time: int, open_: float, high: float, low: float, close: float, volume: int, wap: float, count: int):
    #     # print("RealTimeBar. TickerId:", reqId, RealTimeBar(time, -1, open_, high, low, close, volume, wap, count))
    #     print("TS: {} | Open: {} | High: {} | Low: {} | Close: {} | Volume: {} | Count: {} | VWAP: {}".format(
    #         datetime.datetime.fromtimestamp(time, tz=self.default_tz),
    #         open_,
    #         high,
    #         low,
    #         close,
    #         volume,
    #         count,
    #         wap,
    #     ))

# Making Server Requests


class ApiSocket:
    def __init__(self, wrapper: ApiController):
        self.wrapper = wrapper
        # self.requests = queue.Queue(100)
        # self.results = {}
        self._connected = False
        self._validId = 1
        self._validIdMutex = Lock()

    def connect(self, org, token, url):
        try:
            self.org = org
            self.token = token
            self.url = url
            self.client = InfluxDBClient(url=self.url, token=self.token)
            self.query_api = self.client.query_api()

            self._connected = True
            self._runningRequests = set()
            self._runningOrders: Dict[Contract, Dict[int, Order]] = dict()
            self._runningOrdersMutex = Lock()
            self._cancelOrders = set()
            self._account_updates = False

            self.wrapper.connectAck()
        except Exception as e:
            if self.wrapper:
                self.wrapper.error(-1, 100,
                                   f"Could not connect to database with error message: {e}")
                # self.wrapper.error(NO_VALID_ID, CONNECT_FAIL.code(), CONNECT_FAIL.msg())

    def disconnect(self):
        self._connected = False
        self._runningRequests.clear()
        for orders in self._runningOrders.values():
            for orderId in orders.keys():
                self._cancelOrders.add(orderId)
        wait_until(
            condition_function=lambda: len(self._runningOrders) == 0,
            seconds_to_wait=5,
            msg=f"Waited more than 5 secs to disconnect."
        )
        wait_until(
            condition_function=lambda: sum(1 for name, thread in self.__dict__.items(
            ) if 'thread' in name and thread.is_alive()) == 0,
            seconds_to_wait=5,
            msg=f"Waited more than 5 secs to disconnect."
        )
        print("[Disconnected]")
        return

    def isConnected(self):
        return self._connected

    ########

    def reqContractOrderFill(self, contract: Contract, timeStart: arrow.Arrow):
        # TODO: check for other secType
        if contract.secType == "FUT":
            key = f"{contract.exchange}_{contract.symbol}"
            with open('src/resources/queries/get_futures_historical_bars.influx', 'r') as file:
                query = file.read()
            query = query.replace("v.windowPeriod", "1s") \
                .replace("v.exchange", contract.exchange) \
                .replace("v.symbol", contract.symbol) \
                .replace("v.contract", contract.lastTradeDateOrContractMonth)
        else:
            raise NotImplementedError(f"Order simulation only supports FUT, but got {contract.secType}")

        def row2fill(row, order: Order):
            bar_date = arrow.get(row["_time"]).to(
                ClockController.time_zone).format("YYYYMMDD  HH:mm:ss")
            amount_filled = 0
            if order.action == 'BUY':
                if row["low"] < order.lmtPrice:
                    amount_filled = min(order.totalQuantity, row["volume"]//1)
                elif row["low"] == order.lmtPrice:
                    amount_filled = min(order.totalQuantity, row["volume"]//5)
            elif order.action == 'SELL':
                if row["high"] > order.lmtPrice:
                    amount_filled = min(order.totalQuantity, row["volume"]//1)
                elif row["high"] == order.lmtPrice:
                    amount_filled = min(order.totalQuantity, row["volume"]//5)
            return bar_date, amount_filled

        total_filled = dict()
        status = dict()
        wait_until(
            condition_function=lambda: ClockController.utcnow() > timeStart,
            seconds_to_wait=5,
            msg=f"Waited more than 5 secs for  time to pass."
        )
        while len(self._runningOrders[key]) > 0:
            timeStop = ClockController.utcnow()
            updateQuery = query.replace("v.timeRangeStart", str(timeStart.int_timestamp)) \
                .replace("v.timeRangeStop", str(timeStop.int_timestamp))
            tables = self.query_api.query(updateQuery, org=self.org)
            if len(tables) == 0:
                continue
            for record in tables[0].records:
                for orderId, order in list(self._runningOrders[key].items()):
                    total_filled.setdefault(orderId, 0)
                    status.setdefault(orderId, 'Submitted')
                    bar_date, amount_filled = row2fill(record.values, order)
                    if amount_filled > 0:
                        total_filled[orderId] += amount_filled
                        execution = Execution()
                        execution.orderId = orderId
                        execution.shares = amount_filled
                        execution.cumQty = total_filled[orderId]
                        execution.avgPrice = order.lmtPrice
                        self.wrapper.execDetails(None, None, execution)

                        if total_filled[orderId] == order.totalQuantity:
                            status[orderId] = 'Filled'
                            try:
                                self._runningOrders[key].pop(orderId)
                            except KeyError:
                                self.wrapper.error(
                                    orderId, 102, f"OrderId: {orderId} | Not successfully canceled.")

                    if total_filled[orderId] < order.totalQuantity and orderId in self._cancelOrders:
                        status[orderId] = 'Cancelled'
                        self._cancelOrders.remove(orderId)
                        self._runningOrders[key].pop(orderId)
                        
                    if amount_filled > 0 or status[orderId] == 'Cancelled':
                        self.wrapper.orderStatus(
                            orderId,
                            status[orderId],
                            total_filled[orderId],
                            order.totalQuantity - total_filled[orderId],
                            order.lmtPrice,
                            None, None, None, None, None, None
                        )
                        if status[orderId] == 'Filled':
                            self.wrapper.error(
                                orderId, 300, f"Error: Successfully filled orderId {orderId}.")
                        elif status[orderId] == 'Cancelled':
                            self.wrapper.error(
                                orderId, 300, f"Error: Successfully cancelled orderId {orderId}.")
                        else:
                            self.wrapper.error(
                                orderId, 300, f"Error: Partially filled orderId {orderId} with {amount_filled} shares.")

            timeStart = arrow.get(
                bar_date + " " + ClockController.time_zone, "YYYYMMDD  HH:mm:ss ZZZ")
            wait_until(
                condition_function=lambda: ClockController.utcnow() > timeStart,
                seconds_to_wait=5,
                msg=f"Waited more than 5 secs for  time to pass."
            )
            time.sleep(0.005)

        if self._runningOrdersMutex.acquire():
            self._runningOrders.pop(key)
        self._runningOrdersMutex.release()

    def cancelOrder(self, orderId: int):
        self._cancelOrders.add(orderId)
        try:
            wait_until(
                condition_function=lambda: orderId not in self._cancelOrders,
                seconds_to_wait=1,
                msg=f"Waited more than 1 secs to cancel {orderId}."
            )
        except:
            self._cancelOrders.remove(orderId)
            self.wrapper.error(
                orderId, 102, f"Error: Could not find orderId {orderId} to cancel.")

    def placeOrder(self, orderId: int, contract: Contract, order: Order):
        orderState = OrderState()
        orderState.initMarginChange = "20000"
        orderState.maintMarginChange = "20000"
        orderState.commission = 2.15

        if order.whatIf:
            orderState.status = 'PreSubmitted'
            self.wrapper.openOrder(orderId, contract, order, orderState)
            return

        orderState.status = 'Submitted'
        self.wrapper.openOrder(orderId, contract, order, orderState)
        self.wrapper.orderStatus(
            orderId,
            orderState.status,
            0,
            order.totalQuantity,
            order.lmtPrice,
            None, None, None, None, None, None
        )

        if contract.secType == "FUT":
            key = f"{contract.exchange}_{contract.symbol}"
            if key in self._runningOrders:
                self._runningOrders[key][orderId] = order
            else:
                if self._runningOrdersMutex.acquire():
                    timeStart = ClockController.utcnow()
                    self._runningOrders[key] = {orderId: order}
                    thread = Thread(target=self.reqContractOrderFill,
                                    args=(contract, timeStart))
                    thread.start()
                setattr(self, f"_thread_order_{key}", thread) # overwrite previous thread
                self._runningOrdersMutex.release()
        else:
            raise NotImplementedError(f"Order simulation only supports FUT, but got {contract.secType}")

    ########
    
    def reqAccountUpdates(self):
        accountInfoKeys = [
            'TotalCashBalance',
            'RealizedPnL',
            'UnrealizedPnL'
        ]
        time_counter = 3*60
        while self._account_updates:
            if time_counter >= 3*60:
                time_counter = 0
                # TODO: loops
                # for key in accountInfoKeys:
                #     self.wrapper.updateAccountValue(key, val, 'USD', '')
                # for pos
                # self.wrapper.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, '')
                # self.wrapper.updateAccountTime(timeStamp)
            else:
                time.sleep(1)
                time_counter += 1

    def reqIds(self):
        if self._validIdMutex.acquire(True):
            self.wrapper.nextValidId(self._validId)
            self._validId += 1
        self._validIdMutex.release()

    ########

    def reqContractDetails(self, reqId: int, contract: Contract):
        if contract.secType == "STK":
            contractDetails = ContractDetails()
            contractDetails.contract = copy.deepcopy(contract)
            self.wrapper.contractDetails(reqId, contractDetails)
            self.wrapper.contractDetailsEnd(reqId)

        elif contract.secType == "FUT":
            with open('src/resources/queries/get_futures.influx', 'r') as file:
                query = file.read()
            query = query.replace("v.timeRangeStart", str(ClockController.utcnow().shift(days=-2).int_timestamp)) \
                .replace("v.timeRangeStop", str(ClockController.utcnow().int_timestamp)) \
                .replace("v.windowPeriod", "30s") \
                .replace("v.exchange", contract.exchange) \
                .replace("v.symbol", contract.symbol)

            tables = self.query_api.query(query, org=self.org)
            for table in tables:
                contractDetails = ContractDetails()
                contractDetails.contract = copy.deepcopy(contract)
                contractDetails.contract.lastTradeDateOrContractMonth = table.records[
                    0]["contract"]
                self.wrapper.contractDetails(reqId, contractDetails)
            self.wrapper.contractDetailsEnd(reqId)

        else:
            self.wrapper.error(
                reqId, 101, f"Contract must be of type STK or FUT, but received type: {contract.secType}")

    def reqHistoricalTicks(self, reqId: int, contract: Contract, startDateTime: str,
                           endDateTime: str, numberOfTicks: int = 1000, whatToShow: str = "TRADES", useRth: int = 0,
                           ignoreSize: bool = False, miscOptions: TagValueList = []):
        pass

    def reqHistoricalData(self, reqId: TickerId, contract: Contract, endDateTime: str,
                          durationStr: str, barSizeSetting: str, whatToShow: str,
                          useRTH: int, formatDate: int, keepUpToDate: bool, chartOptions: TagValueList):
        if not keepUpToDate and endDateTime == "":
            raise AttributeError(
                f"reqHistoricalData: If keepUpToDate is False then must specify an endDateTime, but received: {keepUpToDate}, {endDateTime}")
        if whatToShow not in ["TRADES"]:
            raise NotImplementedError(
                f"reqHistoricalData: Only whatToShow=\"TRADES\" is supported, but received:{whatToShow}")
        self._runningRequests.add(reqId)

        barSize2windowPeriod = {
            "1 secs": "1s",
            "5 secs": "5s",
            "10 secs": "10s",
            "15 secs": "15s",
            "30 secs": "30s",
            "1 min": "1m",
            "2 mins": "2m",
            "3 mins": "3m",
            "5 mins": "5m",
            "10 mins": "10m",
            "15 mins": "15m",
            "20 mins": "20m",
            "30 mins": "30m",
            "1 hour": "1h",
            "2 hours": "2h",
            "3 hours": "3h",
            "4 hours": "4h",
            "8 hours": "8h",
            "1 day": "1d"
        }

        if contract.secType == "FUT":
            with open('src/resources/queries/get_futures_historical_bars.influx', 'r') as file:
                query = file.read()
            query = query.replace("v.windowPeriod", barSize2windowPeriod[barSizeSetting]) \
                .replace("v.exchange", contract.exchange) \
                .replace("v.symbol", contract.symbol) \
                .replace("v.contract", contract.lastTradeDateOrContractMonth)
        else:
            # TODO: implement STK
            # self.wrapper.error(
            #     reqId, 101, f"Contract must be of type STK or FUT, but received type: {contract.secType}")
            self.wrapper.error(
                reqId, 101, f"Contract must be of type FUT, but received type: {contract.secType}")

        if keepUpToDate:
            timeStop = ClockController.utcnow()
        else:
            timeStop = arrow.get(endDateTime + " " + ClockController.time_zone)
        if barSizeSetting == "1 secs":
            timeStart = timeStop.shift(minutes=-30)
        elif barSizeSetting == "5 secs":
            timeStart = timeStop.shift(hours=-1)
        elif barSizeSetting == "10 secs":
            timeStart = timeStop.shift(hours=-4)
        elif barSizeSetting == "15 secs":
            timeStart = timeStop.shift(hours=-4)
        elif barSizeSetting == "30 secs":
            timeStart = timeStop.shift(hours=-8)
        elif barSizeSetting == "1 min":
            timeStart = timeStop.shift(days=-1)
        elif barSizeSetting == "2 mins":
            timeStart = timeStop.shift(days=-2)
        elif barSizeSetting == "3 mins":
            timeStart = timeStop.shift(weeks=-1)
        elif barSizeSetting == "5 mins":
            timeStart = timeStop.shift(weeks=-1)
        elif barSizeSetting == "10 mins":
            timeStart = timeStop.shift(weeks=-1)
        elif barSizeSetting == "15 mins":
            timeStart = timeStop.shift(weeks=-1)
        elif barSizeSetting == "20 mins":
            timeStart = timeStop.shift(weeks=-1)
        elif barSizeSetting == "30 mins":
            timeStart = timeStop.shift(weeks=-1)
        elif barSizeSetting == "1 hour":
            timeStart = timeStop.shift(months=-1)
        elif barSizeSetting == "2 hours":
            timeStart = timeStop.shift(months=-1)
        elif barSizeSetting == "3 hours":
            timeStart = timeStop.shift(months=-1)
        elif barSizeSetting == "4 hours":
            timeStart = timeStop.shift(months=-1)
        elif barSizeSetting == "8 hours":
            timeStart = timeStop.shift(months=-1)
        elif barSizeSetting == "1 day":
            timeStart = timeStop.shift(years=-1)

        def row2bar(row):
            bar = BarData()
            bar.date = arrow.get(row["_time"]).to(
                ClockController.time_zone).format("YYYYMMDD  HH:mm:ss")
            bar.open = row["open"]
            bar.high = row["high"]
            bar.low = row["low"]
            bar.close = row["close"]
            bar.volume = row["volume"]
            bar.barCount = row["count"]
            bar.average = row["vwap"]
            return bar

        if reqId not in self._runningRequests:
            self.wrapper.error(
                reqId, 300, f"ReqId: {reqId} | Successfully canceled.")
            return

        histQuery = query.replace("v.timeRangeStart", str(timeStart.int_timestamp)) \
            .replace("v.timeRangeStop", str(timeStop.int_timestamp))
        tables = self.query_api.query(histQuery, org=self.org)
        if len(tables) > 0:
            for record in tables[0].records:
                bar = row2bar(record.values)
                self.wrapper.historicalData(reqId, bar)

        self.wrapper.historicalDataEnd(reqId, timeStart, timeStop)
        self.wrapper.error(
            reqId, 300, f"ReqId: {reqId} | Done fetching historical data.")
        try:
            self._runningRequests.remove(reqId)
        except KeyError:
            self.wrapper.error(
                reqId, 300, f"ReqId: {reqId} | Successfully canceled.")

        if keepUpToDate:
            self._runningRequests.add(reqId)
        while reqId in self._runningRequests:
            timeStart = arrow.get(
                bar.date + " " + ClockController.time_zone, "YYYYMMDD  HH:mm:ss ZZZ")
            timeStop = ClockController.utcnow()
            updateQuery = query.replace("v.timeRangeStart", str(timeStart.int_timestamp)) \
                .replace("v.timeRangeStop", str(timeStop.int_timestamp))
            tables = self.query_api.query(updateQuery, org=self.org)
            for record in tables[0].records:
                bar = row2bar(record.values)
                self.wrapper.historicalData(reqId, bar)
            time.sleep(0.005)

        self.wrapper.error(
            reqId, 300, f"ReqId: {reqId} | Successfully unsubscribed.")

    def cancelHistoricalData(self, reqId: TickerId):
        try:
            self._runningRequests.remove(reqId)
        except KeyError:
            time.sleep(0.005)
            try:
                self._runningRequests.remove(reqId)
            except KeyError:
                self.wrapper.error(
                    reqId, 103, f"Error: Could not find reqId {reqId} to cancel.")


class INFLUX(ApiController, ApiSocket):
    def __init__(self, org, token, url, msgHandler=None):
        ApiController.__init__(self, msgHandler)
        ApiSocket.__init__(self, self)

        self.connect(org, token, url)

    ########

    @iswrapper
    def reqIds(self):
        """
        msgHandler should >>
            utilize: nextValidOrderId = orderId
            init bool: orderIdObtained = False
        """
        thread = Thread(target=super().reqIds,
                        args=())
        thread.start()
        setattr(self, f"_thread_reqIds", thread) # overwrite previous thread

    @iswrapper
    def placeOrder(self, orderId: int, contract: Contract, order: Order):
        """
        msgHandler must >>
            define funcs: handleOpenOrder, handleOrderStatus, handleOrderExecution
        """
        thread = Thread(target=super().placeOrder,
                        args=(orderId, contract, order))
        thread.start()
        setattr(self, f"_thread_order_{orderId}", thread) # overwrite previous thread

    @iswrapper
    def cancelOrder(self, orderId: int):
        """
        """
        thread = Thread(target=super().cancelOrder,
                        args=(orderId,))
        thread.start()
        setattr(self, f"_thread_order_{orderId}", thread) # overwrite previous thread
        
    @iswrapper
    def reqAccountUpdates(self, subscribe:bool, acctCode:str):
        """
        msgHandler must >>
            define funcs: updateAccountData, updatePositionData, updateAccountData
        """
        if subscribe:
            self._account_updates = True
            thread: Thread = getattr(self, f"_thread_accUpd", None)
            if thread is None or not thread.is_alive():
                thread = Thread(target=super().reqAccountUpdates,
                                args=())
                thread.start()
                setattr(self, f"_thread_accUpd", thread) # overwrite previous thread
        elif not subscribe:
            self._account_updates = False            

    @iswrapper
    def reqPositions(self):
        """
        """
        # super().reqPositions(self)
        pass

    @iswrapper
    def cancelPositions(self):
        """
        """
        # super().cancelPositions()
        pass

    ########

    @iswrapper
    def reqContractDetails(self, reqId: int, contract: Contract):
        """
        msgHandler must >>
            init dicts: resolvedContracts, contractDetailsObtained
        """
        thread = Thread(target=super().reqContractDetails,
                        args=(reqId, contract))
        thread.start()
        setattr(self, f"_thread_{reqId}", thread)

    @iswrapper
    def reqHistoricalTicks(self, reqId: int, contract: Contract, startDateTime: str,
                           endDateTime: str, numberOfTicks: int = 1000, whatToShow: str = "TRADES", useRth: int = 0,
                           ignoreSize: bool = False, miscOptions: TagValueList = []):
        """
        msgHandler must >>
            init dicts: resolvedHistoricalTickData, historicalTickDataObtained
        """
        thread = Thread(target=super().reqHistoricalTicks, args=(reqId, contract, startDateTime,
                        endDateTime, numberOfTicks, whatToShow, useRth, ignoreSize, miscOptions))
        thread.start()
        setattr(self, f"_thread_{reqId}", thread)

    @iswrapper
    def reqHistoricalData(self, reqId: TickerId, contract: Contract, endDateTime: str,
                          durationStr: str, barSizeSetting: str, whatToShow: str,
                          useRTH: int, formatDate: int, keepUpToDate: bool, chartOptions: TagValueList):
        """
        msgHandler must >>
            define funcs: updateBarData
            init dicts: resolvedHistoricalBarData
        """
        thread = Thread(target=super().reqHistoricalData, args=(reqId, contract, endDateTime,
                        durationStr, barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate, chartOptions))
        thread.start()
        setattr(self, f"_thread_{reqId}", thread)

    @iswrapper
    def cancelHistoricalData(self, reqId: TickerId):
        """
        msgHandler must >>
        """
        thread = Thread(target=super().cancelHistoricalData, args=(reqId,))
        thread.start()
        setattr(self, f"_thread_X_{reqId}", thread)
