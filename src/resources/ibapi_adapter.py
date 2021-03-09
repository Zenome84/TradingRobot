
import pytz
import time
import arrow

from threading import Thread

from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.utils import iswrapper
# from ibapi.order import Order
from ibapi.contract import Contract, ContractDetails
from ibapi.common import ListOfHistoricalTickLast, TagValueList, BarData, TickerId
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

    @iswrapper
    def error(self, reqId, errorCode, errorString):
        print("Error. Id: ", reqId, " Code: ",
              errorCode, " Msg: ", errorString)

    @iswrapper
    def connectAck(self):
        print("\n[Connected]")
        time.sleep(0.1)

    ###########################################

    # reqContractDetails
    @iswrapper
    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        if self.msgHandler is not None:
            self.msgHandler.resolvedContracts.setdefault(
                reqId, []).append(contractDetails.contract)
        else:
            print("Contract Details: {}".format(contractDetails.contract))
            self.resolved_contract.append(contractDetails.contract)

    @iswrapper
    def contractDetailsEnd(self, reqId: int):
        if self.msgHandler is not None:
            self.msgHandler.contractDetailsObtained[reqId] = True
        else:
            print("Completed Request: {}".format(reqId))
            self.contractDetailsIsObtained = True

    ###########################################

    # reqHistoricalData
    @iswrapper
    def historicalData(self, reqId: int, bar: BarData):
        if self.msgHandler is not None:
            self.msgHandler.updateBarData(reqId, bar)
        else:
            # print("HistoricalData. ReqId:", reqId, "BarData.", bar)
            print("TS: {} | Open: {} | High: {} | Low: {} | Close: {} | Volume: {} | Count: {} | VWAP: {}".format(
                arrow.get(bar.date + " " + str(self.default_tz), "YYYYMMDD  HH:mm:ss ZZZ"),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.barCount,
                bar.average,
            ))

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)

    @iswrapper
    def historicalDataUpdate(self, reqId: int, bar: BarData):
        # print("HistoricalDataUpdate. ReqId:", reqId, "BarData.", bar)
        print("TS: {} | Open: {} | High: {} | Low: {} | Close: {} | Volume: {} | Count: {} | VWAP: {}".format(
            arrow.get(bar.date + " " + str(self.default_tz), "YYYYMMDD  HH:mm:ss ZZZ"),
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            bar.barCount,
            bar.average,
        ))

    ###########################################

    # reqHistoricalTicks for "TRADES" only
    @iswrapper
    def historicalTicksLast(self, reqId: int, ticks: ListOfHistoricalTickLast, done: bool):
        if self.msgHandler is not None:
            self.msgHandler.resolvedHistoricalTickData[reqId] = ticks
            self.msgHandler.historicalTickDataObtained[reqId] = True
        else:
            for tick in ticks:
                print("HistoricalTickLast. Time: {} | PastLimit: {} | Unreported: {} | Price: {} | Size: {} | Exchange: {} | SpecialConditions: {}".format(
                    arrow.get(tick.time).to(self.default_tz),
                    tick.tickAttribLast.pastLimit,
                    tick.tickAttribLast.unreported,
                    tick.price,
                    tick.size,
                    tick.exchange,
                    tick.specialConditions
                ))
            print("Completed Request: {} | Count: {}".format(reqId, len(ticks)))

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
    def reqContractDetails(self, reqId: int, contract: Contract):
        """
        msgHandler must init dicts: resolvedContracts, contractDetailsObtained
        """
        super().reqContractDetails(reqId, contract)

    @iswrapper
    def reqHistoricalTicks(self, reqId: int, contract: Contract, startDateTime: str,
                           endDateTime: str, numberOfTicks: int = 1000, whatToShow: str = "TRADES", useRth: int = 0,
                           ignoreSize: bool = False, miscOptions: TagValueList = []):
        """
        msgHandler must init dicts: resolvedHistoricalTickData, historicalTickDataObtained
        """
        super().reqHistoricalTicks(reqId, contract, startDateTime, endDateTime, numberOfTicks,
                                   whatToShow, useRth, ignoreSize, miscOptions)

    @iswrapper
    def reqHistoricalData(self, reqId: TickerId, contract: Contract, endDateTime: str,
                          durationStr: str, barSizeSetting: str, whatToShow: str,
                          useRTH: int, formatDate: int, keepUpToDate: bool, chartOptions: TagValueList):
        """
        msgHandler must define: updateBarData
        """
        super().reqHistoricalData(reqId, contract, endDateTime, durationStr,
                                  barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate, chartOptions)


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
    # # contract.primaryExchange = "ARCA"
    # contract.currency = "USD"

    # paper = True
    port = 7496
    # if paper == True:
    # port = 7497

    tApp = IBAPI('127.0.0.1', port, 0)
    timePassed = 0
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

    tApp.reqHistoricalData(4, front_contract, "", "1 D",
                           "1 min", "TRADES", 1, 1, True, [])
    timePassed = 0
    while True:
        time.sleep(0.1)
        timePassed += 0.1
        if timePassed > 125:
            raise RuntimeError(
                "Waited more than 5 secs to get contract details")
                    

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
