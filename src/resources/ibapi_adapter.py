
from PIL.Image import new
from ibapi.execution import Execution
from ibapi.order import Order
from ibapi.order_state import OrderState
import pytz
import time
import arrow
from decimal import Decimal

from threading import Thread

from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.utils import iswrapper
# from ibapi.order import Order
from ibapi.contract import Contract, ContractDetails
from ibapi.common import ListOfHistoricalTickLast, OrderId, TagValueList, BarData, TickerId
from ibapi.ticktype import TickTypeEnum

from resources.time_tools import ClockController


class ApiController(EWrapper):
    def __init__(self, msgHandler=None, time_zone=ClockController.time_zone):
        EWrapper.__init__(self)
        self.msgHandler = msgHandler
        self.default_tz = pytz.timezone(time_zone)

        if msgHandler is None:
            self.resolved_contract = []
            self.contractDetailsIsObtained = False
            self.orders = {}

    @iswrapper
    def error(self, reqId, errorCode, errorString):
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
                f"Asset: {contract.symbol}@{contract.exchange} | " +
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

    # @iswrapper
    # def accountDownloadEnd(self, accountName: str):
    #     super().accountDownloadEnd(accountName)
    #     print("AccountDownloadEnd. Account:", accountName)

    ###########################################

    # @iswrapper
    # def accountSummary(self, reqId: int, account: str, tag: str, value: str,
    #                    currency: str):
    #     super().accountSummary(reqId, account, tag, value, currency)
    #     print("AccountSummary. ReqId:", reqId, "Account:", account,
    #           "Tag: ", tag, "Value:", value, "Currency:", currency)
    
    # @iswrapper
    # def accountSummaryEnd(self, reqId: int):
    #     super().accountSummaryEnd(reqId)
    #     print("AccountSummaryEnd. ReqId:", reqId)

    ###########################################
    
    # @iswrapper
    # def position(self, account: str, contract: Contract, position: Decimal,
    #              avgCost: float):
    #     if self.msgHandler is not None:
    #         self.msgHandler.updatePositionData(contract, position)
    #     else:
    #         print(
    #             f"Account: {account} | " +
    #             f"Asset: {contract.symbol}@{contract.exchange} | " +
    #             f"Position: {position} | "
    #         )

    # @iswrapper
    # def positionEnd(self):
    #     super().positionEnd()
    #     print("PositionEnd")

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
            self.lastPrice = bar.close

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        if self.msgHandler is not None:
            self.msgHandler.resolvedHistoricalBarData[reqId] = True
        else:
            super().historicalDataEnd(reqId, start, end)
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


class ApiSocket(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


class IBAPI(ApiController, ApiSocket):
    def __init__(self, ipAddress, portId, clientId, msgHandler=None):
        ApiController.__init__(self, msgHandler)
        ApiSocket.__init__(self, wrapper=self)

        self.connect(ipAddress, portId, clientId)

        thread = Thread(target=self.run)
        thread.start()
        setattr(self, "_thread", thread)

    @iswrapper
    def reqIds(self):
        """
        msgHandler should >>
            utilize: nextValidOrderId = orderId
            init bool: orderIdObtained = False
        """
        super().reqIds(-1)

    @iswrapper
    def placeOrder(self, orderId: int, contract: Contract, order: Order):
        """
        msgHandler must >>
            define funcs: handleOpenOrder, handleOrderStatus, handleOrderExecution
        """
        super().placeOrder(orderId, contract, order)

    @iswrapper
    def cancelOrder(self, orderId: int):
        """
        """
        super().cancelOrder(orderId)
        
    @iswrapper
    def reqAccountUpdates(self, subscribe:bool, acctCode:str):
        """
        msgHandler must >>
            define funcs: updateAccountData, updatePositionData, updateAccountData
        """
        super().reqAccountUpdates(subscribe, acctCode)

    # @iswrapper
    # def reqPositions(self):
    #     """
    #     """
    #     super().reqPositions(self)

    # @iswrapper
    # def cancelPositions(self):
    #     """
    #     """
    #     super().cancelPositions()

    @iswrapper
    def reqContractDetails(self, reqId: int, contract: Contract):
        """
        msgHandler must >>
            init dicts: resolvedContracts, contractDetailsObtained
        """
        super().reqContractDetails(reqId, contract)

    @iswrapper
    def reqHistoricalTicks(self, reqId: int, contract: Contract, startDateTime: str,
                           endDateTime: str, numberOfTicks: int = 1000, whatToShow: str = "TRADES", useRth: int = 0,
                           ignoreSize: bool = False, miscOptions: TagValueList = []):
        """
        msgHandler must >>
            init dicts: resolvedHistoricalTickData, historicalTickDataObtained
        """
        super().reqHistoricalTicks(reqId, contract, startDateTime, endDateTime, numberOfTicks,
                                   whatToShow, useRth, ignoreSize, miscOptions)

    @iswrapper
    def reqHistoricalData(self, reqId: TickerId, contract: Contract, endDateTime: str,
                          durationStr: str, barSizeSetting: str, whatToShow: str,
                          useRTH: int, formatDate: int, keepUpToDate: bool, chartOptions: TagValueList):
        """
        msgHandler must >>
            define funcs: updateBarData
            init dicts: resolvedHistoricalBarData
        """
        super().reqHistoricalData(reqId, contract, endDateTime, durationStr,
                                  barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate, chartOptions)

    @iswrapper
    def cancelHistoricalData(self, reqId: TickerId):
        """
        msgHandler must >>
        """
        super().cancelHistoricalData(reqId)


if __name__ == "__main__":
    # Create a new contract object for the E-mini NASDAQ-100
    contract = Contract()
    contract.symbol = "ES"
    contract.secType = "FUT"
    contract.exchange = "GLOBEX"
    contract.currency = "USD"
    contract.includeExpired = True

    # contract.symbol = "SPY"
    # contract.secType = "STK"
    # contract.exchange = "SMART"
    # # contract.exchange = "ARCA"
    # contract.currency = "USD"

    # paper = True
    # port = 7496
    # if paper == True:
    port = 7497

    tApp = IBAPI('127.0.0.1', port, 0)
    timePassed = 0.
    while not(tApp.isConnected()):
        time.sleep(0.1)
        timePassed += 0.1
        if timePassed > 5:
            raise RuntimeError(
                "Waited more than 5 secs to establish connection")

    tApp.contractDetailsIsObtained = False
    tApp.reqContractDetails(reqId=2, contract=contract)
    timePassed = 0
    while not(tApp.contractDetailsIsObtained):
        time.sleep(0.1)
        timePassed += 0.1
        if timePassed > 5:
            raise RuntimeError(
                "Waited more than 5 secs to get contract details")

    max_month = 20300000
    for contract in tApp.resolved_contract:
        # and contract.conId != 299552802:
        if int(contract.lastTradeDateOrContractMonth) < max_month \
                and int(contract.lastTradeDateOrContractMonth) > int(arrow.utcnow().format("YYYYMMDD")):
            max_month = int(contract.lastTradeDateOrContractMonth)
            front_contract = contract

    # tApp.reqHistoricalData(4, front_contract, "", "1 D",
    #                        "1 min", "TRADES", 0, 1, True, [])
    time.sleep(1)
    # tApp.reqAccountSummary(9003, "All", "$LEDGER:EUR")
    tApp.reqAccountUpdates(True, '')
    tApp.reqPositions()
    time.sleep(2)
    from resources.ibapi_orders import Orders
    # orders = Orders.BracketOrder(tApp.nextValidOrderId, "BUY", 1, 4582.00, 4582.5, 4581.5)
    orders = [
        Orders.LimitOrder(tApp.nextValidOrderId, "BUY", 1, 4525)
    ]
    for order in orders:
        order.whatIf = True
        tApp.placeOrder(order.orderId, front_contract, order)
        time.sleep(1)
        tApp.cancelOrder(order.orderId)
    timePassed = 0
    break_loop = False
    while not break_loop:
        time.sleep(0.1)
        timePassed += 0.1
        if timePassed > 5:
            # raise RuntimeError(
            #     "Waited more than 5 secs to get contract details")
            tApp.reqAccountUpdates(False, '')
            tApp.cancelPositions()
            break_loop = True
    time.sleep(1)
    tApp.disconnect()
    time.sleep(1)
    exit(0)
    

    # quit_app = False
    # while not quit_app:
    #     time.sleep(0.5)
    #     for order in orders:
    #         newPrice = round(4*(0.999 if order.action == "BUY" else 1.001)*tApp.lastPrice)/4
    #         if abs(newPrice- order.lmtPrice) >= 0.5:
    #             order.lmtPrice = newPrice
    #             tApp.placeOrder(order.orderId, front_contract, order)
    #     timePassed += 0.5
    #     if timePassed > 150:
    #         for order in orders:
    #             tApp.cancelOrder(order.orderId)
    #         quit_app = True

    time.sleep(1)
    tApp.disconnect()
    time.sleep(1)

# OpenOrder. PermId:  1027172670 ClientId: 0  OrderId: 19 Account: DU2870980 Symbol: ES SecType: FUT Exchange: GLOBEX Action: BUY OrderType: LMT TotalQty: 1.0 CashQty: 0.0 LmtPrice: 4582.0 AuxPrice: 0.0 Status: Submitted
# OrderStatus. Id: 19 Status: Submitted Filled: 0.0 Remaining: 1.0 AvgFillPrice: 0.0 PermId: 1027172670 ParentId: 0 LastFillPrice: 0.0 ClientId: 0 WhyHeld:  MktCapPrice: 0.0

# OpenOrder. PermId:  1027172671 ClientId: 0  OrderId: 20 Account: DU2870980 Symbol: ES SecType: FUT Exchange: GLOBEX Action: SELL OrderType: LMT TotalQty: 1.0 CashQty: 0.0 LmtPrice: 4582.5 AuxPrice: 0.0 Status: PreSubmitted
# OrderStatus. Id: 20 Status: PreSubmitted Filled: 0.0 Remaining: 1.0 AvgFillPrice: 0.0 PermId: 1027172671 ParentId: 19 LastFillPrice: 0.0 ClientId: 0 WhyHeld: child MktCapPrice: 0.0

# OpenOrder. PermId:  1027172672 ClientId: 0  OrderId: 21 Account: DU2870980 Symbol: ES SecType: FUT Exchange: GLOBEX Action: SELL OrderType: STP TotalQty: 1.0 CashQty: 0.0 LmtPrice: 0.0 AuxPrice: 4581.5 Status: PreSubmitted
# OrderStatus. Id: 21 Status: PreSubmitted Filled: 0.0 Remaining: 1.0 AvgFillPrice: 0.0 PermId: 1027172672 ParentId: 19 LastFillPrice: 0.0 ClientId: 0 WhyHeld: child,trigger MktCapPrice: 0.0


# OpenOrder. PermId:  1027172672 ClientId: 0  OrderId: 21 Account: DU2870980 Symbol: ES SecType: FUT Exchange: GLOBEX Action: SELL OrderType: STP TotalQty: 1.0 CashQty: 0.0 LmtPrice: 0.0 AuxPrice: 4581.5 Status: PreSubmitted
# OrderStatus. Id: 21 Status: PreSubmitted Filled: 0.0 Remaining: 1.0 AvgFillPrice: 0.0 PermId: 1027172672 ParentId: 19 LastFillPrice: 0.0 ClientId: 0 WhyHeld: child,trigger MktCapPrice: 0.0

# OpenOrder. PermId:  1027172672 ClientId: 0  OrderId: 21 Account: DU2870980 Symbol: ES SecType: FUT Exchange: GLOBEX Action: SELL OrderType: STP TotalQty: 1.0 CashQty: 0.0 LmtPrice: 0.0 AuxPrice: 4581.5 Status: PreSubmitted
# OrderStatus. Id: 21 Status: PreSubmitted Filled: 0.0 Remaining: 1.0 AvgFillPrice: 0.0 PermId: 1027172672 ParentId: 19 LastFillPrice: 0.0 ClientId: 0 WhyHeld: child,trigger MktCapPrice: 0.0

# OpenOrder. PermId:  1027172672 ClientId: 0  OrderId: 21 Account: DU2870980 Symbol: ES SecType: FUT Exchange: GLOBEX Action: SELL OrderType: STP TotalQty: 1.0 CashQty: 0.0 LmtPrice: 0.0 AuxPrice: 4581.5 Status: PreSubmitted
# OrderStatus. Id: 21 Status: PreSubmitted Filled: 0.0 Remaining: 1.0 AvgFillPrice: 0.0 PermId: 1027172672 ParentId: 19 LastFillPrice: 0.0 ClientId: 0 WhyHeld: child,trigger MktCapPrice: 0.0


    print('Done')

#     # # endDateTime =  arrow.get(front_contract.lastTradeDateOrContractMonth + " 12:00:00").shift(days=-3)
#     # # startDateTime = endDateTime.shift(seconds=-5)
#     # tApp.reqHistoricalTicks(4, tApp.resolved_contract[0],
#     #                         "",
#     #                         str(int(
#     #                             front_contract.lastTradeDateOrContractMonth)-0) + " 08:59:58",
#     #                         # "20210129 16:00:00",
#     #                         1000,
#     #                         "TRADES",
#     #                         0,
#     #                         False,
#     #                         []
#     #                         )
#     # time.sleep(5)

#     # # tApp.reqMktData(1, front_contract, "", False, False, [])
#     # # tApp.reqHistoricalData(1, front_contract, front_contract.lastTradeDateOrContractMonth + " 00:00:00", "120 S", "5 secs", "TRADES", 0, 2, False, [])
#     # tApp.reqHistoricalData(
#     #     1, tApp.resolved_contract[0],  "20210130 05:00:00", "1 D", "1 min", "TRADES", 0, 2, False, [])
#     # # tApp.reqRealTimeBars(2, front_contract, 5, "TRADES", False, [])
#     # time.sleep(20)

#     tApp.disconnect()
#     timePassed = 0
#     time.sleep(0.1)
#     # while tApp.isConnected():
#     #     time.sleep(0.1)
#     #     timePassed += 0.1
#     #     if timePassed > 5:
#     #         raise RuntimeError("Waited more than 5 secs to disconnect")
#     print('Done')
