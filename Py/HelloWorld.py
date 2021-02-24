import time

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errodCode, errorString):
        print("Error: ", reqId, " ", errodCode, " ", errorString)

    def contractDetails(self, reqId, contractDetails):
        print("contractDetails: ", reqId, " ", contractDetails)


def main():
    app = TestApp()

    app.connect("127.0.0.1", 7496, 0)
    time.sleep(1)

    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"

    app.reqContractDetails(1, contract)

    app.run()


if __name__ == "__main__":
    main()
