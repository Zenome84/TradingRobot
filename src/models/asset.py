
import time
import arrow

# from enum import Enum
# from threading import Lock, Thread
# from resources.ibapi_adapter import *
from ibapi.contract import Contract, ContractDetails
from resources.time_tools import wait_until, ClockController


class Asset(Contract):

    def __init__(self, symbol, exchange, secType, client):
        '''
        a wrapper/collection to store all Signals related to one Contract
            secType is either 'FUT' or 'STK'
        '''
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
            msg="Waited more than 5 secs to get contract details for: {}@{}".format(symbol, exchange)
        )

        self.client.contractDetailsObtained.pop(reqId)
        contracts = self.client.resolvedContracts.pop(reqId)
        self.client.reqIds.remove(reqId)

        if secType == 'FUT':
            date_now = int(arrow.utcnow().to(ClockController.time_zone).shift(days=-1).format("YYYYMMDD"))
            max_month = 30000000
            for contract in contracts:
                contract_month = int(contract.lastTradeDateOrContractMonth)
                if date_now < contract_month < max_month:
                    max_month = contract_month
                    self.contract = contract
        else:
            self.contract = contracts[0]

    def subscribe_bar_signal(self):
        pass
