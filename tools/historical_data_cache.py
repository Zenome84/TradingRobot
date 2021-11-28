
import os
import gc
import json
import glob
import math
import arrow
import random
from enum import Enum
from threading import Lock, Thread
from resources.ibapi_adapter import *


class DataDownloader:
    class ContractType(Enum):
        STK = 1
        FUT = 2

    def __init__(self, connect=True):
        self.time_zone = 'US/Eastern'
        self.live = True
        self.clientId = 1

        if connect:
            self.connect_client()

        self.reqIds = set([0])
        self.reqIds_mutex = Lock()

        self.resolvedContracts = dict()
        self.contractDetailsObtained = dict()
        self.resolvedHistoricalTickData = dict()
        self.historicalTickDataObtained = dict()
        self.resolvedHistoricalBarData = dict()

        self.barTypesMinDuration = {
            '1 sec': '1800 S',  # 30 mins
            '5 sec': '3600 S',  # 1 hr
            '10 sec': '14400 S',  # 4 hrs
            '30 sec': '28800 S',  # 8 hrs
            '1 min': '1 D',
            '2 mins': '2 D',
            '3 mins': '1 W',
            '5 mins': '1 W',
            '10 mins': '1 W',
            '15 mins': '1 W',
            '20 mins': '1 W',
            '30 mins': '1 M',
            '1 hr': '1 M',
            '2 hrs': '1 M',
            '3 hrs': '1 M',
            '4 hrs': '1 M',
            '8 hrs': '1 M',
            '1 day': '1 Y'
        }

        self.contractCache = dict()
        self.tickDataCache = dict()
        self.barDataCache = dict()

    def connect_client(self):
        self.tws_client = IBAPI(
            '127.0.0.1', 7496 if self.live else 7497, self.clientId, self)
        timePassed = 0
        while not self.tws_client.isConnected():
            time.sleep(0.1)
            timePassed += 0.1
            if timePassed > 5:
                raise RuntimeError(
                    "Waited more than 5 secs to establish connection")

    def disconnect_client(self):
        self.tws_client.disconnect()
        self.tws_client = None

    def client_connected(self):
        return self.tws_client is not None and self.tws_client.isConnected()

    def get_new_reqIds(self, n=1):
        if self.reqIds_mutex.acquire():
            next_reqId = max(self.reqIds) + 1
            new_reqIds = list(range(next_reqId, next_reqId + n))
            self.reqIds.add(*new_reqIds)
        self.reqIds_mutex.release()
        return new_reqIds

    def bufferStock(self, symbol, exchange):
        contract = Contract()
        contract.symbol = symbol
        contract.exchange = exchange

        contract.secType = 'STK'
        contract.currency = 'USD'

        self._bufferContract(contract)

    def bufferFutures(self, symbol, exchange):
        contract = Contract()
        contract.symbol = symbol
        contract.exchange = exchange
        contract.includeExpired = True

        contract.secType = 'FUT'
        contract.currency = 'USD'

        self._bufferContract(contract)

    def _bufferContract(self, contract):
        if not self.client_connected():
            self.connect_client()

        self.tws_client.contractDetailsIsObtained = False

        reqId = self.get_new_reqIds()[0]
        self.contractDetailsObtained[reqId] = False
        self.tws_client.reqContractDetails(reqId=reqId, contract=contract)
        timePassed = 0
        while not self.contractDetailsObtained[reqId]:
            time.sleep(0.1)
            timePassed += 0.1
            if timePassed > 5:
                raise RuntimeError(
                    "Waited more than 5 secs to get contract details")
        self.contractDetailsObtained.pop(reqId)
        self.contractCache["{}_{}".format(
            contract.symbol, contract.exchange)] = self.resolvedContracts.pop(reqId)
        self.reqIds.remove(reqId)

    def bufferHistoricalTickData(self, symbol, exchange, start_ts, end_ts, contractType=ContractType.STK):
        cache_key = "{}_{}".format(symbol, exchange)

        if cache_key not in self.contractCache:
            if contractType is self.ContractType.STK:
                self.bufferStock(*cache_key)
            elif contractType is self.ContractType.FUT:
                self.bufferFutures(*cache_key)
            else:
                raise RuntimeError(
                    "Unrecognized contract type: {}".format(contractType))

        if not self.client_connected():
            self.connect_client()

        self.tickDataCache[cache_key] = dict()
        cache_path = "U:/MarketData/{}".format(cache_key)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        threads = []
        for contract in self.contractCache[cache_key]:
            if contractType is self.ContractType.STK:
                contract_end_ts = end_ts
                contract_start_ts = start_ts
                contract_key = 0
            elif contractType is self.ContractType.FUT:
                contract_end_ts = arrow.get(contract.lastTradeDateOrContractMonth +
                                            " 09:30:00 " + self.time_zone, "YYYYMMDD HH:mm:ss ZZZ").int_timestamp
                contract_start_ts = max(
                    start_ts, contract_end_ts - 8640000)  # 100 days
                contract_end_ts = min(end_ts, contract_end_ts)
                contract_key = int(contract.lastTradeDateOrContractMonth)

            if contract_end_ts <= contract_start_ts:
                continue

            self.tickDataCache[cache_key][contract_key] = dict()
            contract_path = "{}/{}".format(cache_path, contract_key)
            if not os.path.exists(contract_path):
                os.makedirs(contract_path)
            contract_files = sorted(
                glob.glob(contract_path + "/*.json"), reverse=True)
            for contract_file in contract_files:
                ymd = contract_file[-13:-5]
                yr = int(ymd[:4])
                mt = int(ymd[4:6])
                dy = int(ymd[6:8])
                if yr not in self.tickDataCache[cache_key][contract_key].keys():
                    self.tickDataCache[cache_key][contract_key][yr] = {
                        mt: {dy: contract_file}}
                elif mt not in self.tickDataCache[cache_key][contract_key][yr].keys():
                    self.tickDataCache[cache_key][contract_key][yr][mt] = {
                        dy: contract_file}
                else:
                    self.tickDataCache[cache_key][contract_key][yr][mt][dy] = contract_file

            threads.append(Thread(target=self._bufferContractHistoricalTickData, args=(
                contract, contract_start_ts, contract_end_ts, cache_key, contract_key, contract_path)))
            threads[-1].start()

            running = 3
            while running > 2:
                running = 0
                for thread in threads:
                    time.sleep(1)
                    if thread.is_alive():
                        running += 1

        running = 1
        while running > 0:
            running = 0
            for thread in threads:
                time.sleep(1)
                if thread.is_alive():
                    running += 1

    def _bufferContractHistoricalTickData(self, contract, contract_start_ts, contract_end_ts, cache_key, contract_key, contract_path):

        daily_dict = dict()
        curr_ts = contract_end_ts
        while curr_ts > contract_start_ts:
            old_ts = curr_ts
            req_ts = arrow.get(
                curr_ts-1).to(self.time_zone).format("YYYYMMDD HH:mm:ss")
            yr = int(req_ts[:4])
            mt = int(req_ts[4:6])
            dy = int(req_ts[6:8])
            req_ts = arrow.get(curr_ts).to(
                self.time_zone).format("YYYYMMDD HH:mm:ss")
            if yr in self.tickDataCache[cache_key][contract_key].keys() \
                    and mt in self.tickDataCache[cache_key][contract_key][yr].keys() \
                    and dy in self.tickDataCache[cache_key][contract_key][yr][mt].keys():
                curr_ts = arrow.get("{:04d}{:02d}{:02d} 00:00:00 {}".format(
                    yr, mt, dy, self.time_zone), "YYYYMMDD HH:mm:ss ZZZ").int_timestamp
                daily_dict = dict()
                continue

            reqId = self.get_new_reqIds()[0]
            self.historicalTickDataObtained[reqId] = False
            self.tws_client.reqHistoricalTicks(reqId=reqId, contract=contract,
                                               startDateTime="", endDateTime=req_ts)
            timePassed = 0
            while not self.historicalTickDataObtained[reqId]:
                time.sleep(0.1)
                timePassed += 0.1
                if timePassed > 120:
                    raise RuntimeError(
                        "Waited more than 120 secs to get historical tick data for contract: {} | ts: {}".format(contract, curr_ts))
            self.historicalTickDataObtained.pop(reqId)
            for tick in reversed(self.resolvedHistoricalTickData.pop(reqId)):
                if tick.time < curr_ts:
                    curr_ts = tick.time
                    tick_idx = 9999
                else:
                    tick_idx -= 1
                arrow_ts = arrow.get(tick.time).to(self.time_zone)
                time_nyc = arrow_ts.format("YYYYMMDD HH:mm:ss")
                if int(time_nyc[6:8]) != dy:
                    if len(daily_dict) > 0:
                        contract_file = "{}/{:04d}{:02d}{:02d}.json".format(
                            contract_path, yr, mt, dy)
                        if yr not in self.tickDataCache[cache_key][contract_key].keys():
                            self.tickDataCache[cache_key][contract_key][yr] = {
                                mt: {dy: contract_file}}
                        elif mt not in self.tickDataCache[cache_key][contract_key][yr].keys():
                            self.tickDataCache[cache_key][contract_key][yr][mt] = {
                                dy: contract_file}
                        else:
                            self.tickDataCache[cache_key][contract_key][yr][mt][dy] = contract_file
                        with open(contract_file, 'w') as f:
                            json.dump(daily_dict, f)
                        daily_dict = dict()
                        gc.collect()
                    yr = int(time_nyc[:4])
                    mt = int(time_nyc[4:6])
                    dy = int(time_nyc[6:8])

                daily_dict[tick.time*10000 + tick_idx] = {
                    "time_nyc": time_nyc,
                    "time_ts": tick.time,
                    "tick_idx": tick_idx,
                    "past_limit": tick.tickAttribLast.pastLimit,
                    "unreported": tick.tickAttribLast.unreported,
                    "price": tick.price,
                    "size": tick.size,
                    "exchange": tick.exchange,
                    "special_condition": tick.specialConditions
                }
            self.reqIds.remove(reqId)
            print("Buffered {:.2%} of Tick Data for {} {}".format(
                (contract_end_ts - curr_ts)/(contract_end_ts - contract_start_ts), cache_key, contract_key))
            if old_ts == curr_ts:
                curr_ts -= 1

        if len(daily_dict) > 0:
            contract_file = "{}/{:04d}{:02d}{:02d}.json".format(
                contract_path, yr, mt, dy)
            if yr not in self.tickDataCache[cache_key][contract_key].keys():
                self.tickDataCache[cache_key][contract_key][yr] = {
                    mt: {dy: contract_file}}
            elif mt not in self.tickDataCache[cache_key][contract_key][yr].keys():
                self.tickDataCache[cache_key][contract_key][yr][mt] = {
                    dy: contract_file}
            else:
                self.tickDataCache[cache_key][contract_key][yr][mt][dy] = contract_file
            with open(contract_file, 'w') as f:
                json.dump(daily_dict, f)

    # def bufferHistoricalData(self, symbol, exchange):
    #     if not self.client_connected():
    #         self.connect_client()

    #     for contract in self.contractCache[(symbol, exchange)]:
    #         # arrow.get(contract.lastTradeDateOrContractMonth + " 00:00:00").datetime


if __name__ == "__main__":

    dd = DataDownloader()
    # dd.bufferFutures('ES', 'GLOBEX')
    # dd.bufferStock('SPY', 'SMART')
    dd.bufferFutures('NQ', 'GLOBEX')
    dd.bufferHistoricalTickData('NQ', 'GLOBEX',
                                arrow.get("20181216 18:30:00 " + dd.time_zone,
                                          "YYYYMMDD HH:mm:ss ZZZ").int_timestamp,
                                arrow.get("20210723 05:00:00 " + dd.time_zone,
                                          "YYYYMMDD HH:mm:ss ZZZ").int_timestamp,
                                dd.ContractType.FUT)
    dd.disconnect_client()
    print("Done")
