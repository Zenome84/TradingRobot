
import time
import arrow

# from enum import Enum
# from threading import Lock, Thread
# from resources.ibapi_adapter import *
from ibapi.contract import Contract, ContractDetails

from resources.enums import BarDuration, BarSize
from resources.time_tools import wait_until, ClockController
from models.signal import Signal


class Asset(Contract):

    def __init__(self, symbol, exchange, secType, client):
        '''
        a wrapper/collection to store all Signals related to one Contract
            secType is either 'FUT' or 'STK'
        '''
        if secType not in ['FUT', 'STK']:
            raise RuntimeError("secType must be 'FUT' or 'STK', but received: {}".format(secType))

        self.client = client
        self.signals = dict()

        self.contract = Contract()
        self.contract.symbol = symbol
        self.contract.exchange = exchange
        self.contract.currency = 'USD'
        self.contract.secType = secType

        if not self.client.client_connected():
            self.client.connect_client()

        reqId = self.client.get_new_reqIds()[0]
        self.client.contractDetailsObtained[reqId] = False
        self.client.tws_client.reqContractDetails(
            reqId=reqId, contract=self.contract)

        wait_until(
            condition_function=lambda: self.client.contractDetailsObtained[reqId],
            seconds_to_wait=5,
            msg="Waited more than 5 secs to get contract details for: {}@{}".format(
                symbol, exchange)
        )

        self.client.contractDetailsObtained.pop(reqId)
        contracts = self.client.resolvedContracts.pop(reqId)
        self.client.resolve_request(reqId)

        if secType == 'FUT':
            date_now = int(arrow.utcnow().to(
                ClockController.time_zone).shift(days=-1).format("YYYYMMDD"))
            max_month = 30000000
            for contract in contracts:
                contract_month = int(contract.lastTradeDateOrContractMonth)
                if date_now < contract_month < max_month:
                    max_month = contract_month
                    self.contract = contract
        else:
            self.contract = contracts[0]
        
        self.asset_key = "{}@{}".format(symbol, exchange)

    def subscribe_bar_signal(self, bar_size, length):
        if not self.client.client_connected():
            self.client.connect_client()

        reqId = self.client.get_new_reqIds()[0]
        self.signals[reqId] = Signal(
            length, "{}@{}:{}".format(self.contract.symbol, self.contract.exchange, bar_size.name))
        self.client.resolvedHistoricalBarData[reqId] = False
        self.client.request_signal_map[reqId] = self.asset_key
        self.client.tws_client.reqHistoricalData(
            reqId, self.contract, "", BarDuration[bar_size.value], bar_size.value, "TRADES", 0, 1, True, [])

        wait_until(
            condition_function=lambda: self.client.resolvedHistoricalBarData[reqId],
            seconds_to_wait=15,
            msg="Waited more than 15 secs to get {} bars for: {}@{}".format(
                bar_size.value, self.contract.symbol, self.contract.exchange)
        )

        return reqId

    def unsubscribe_bar_signal(self, reqId):
        self.client.tws_client.cancelHistoricalData(reqId)
        self.client.request_signal_map.pop(reqId)
        self.client.resolve_request(reqId)
        self.signals.pop(reqId)

    def updateBarData(self, reqId, bar):
        self.signals[reqId].updateData([
            arrow.get(bar.date + " " + str(ClockController.time_zone), "YYYYMMDD  HH:mm:ss ZZZ").timestamp(),
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            0 if bar.volume == -1 else bar.volume,
            0 if bar.barCount == -1 else bar.barCount,
            bar.average,
        ])
        