
from dataclasses import dataclass
import datetime
from tkinter.messagebox import NO
from typing import List, Dict, Optional, Set
import arrow
import time
from core.robot_client import RobotClient
from models.signal import Signal
from resources.enums import BarSize
from resources.time_tools import ClockController, wait_until



class Alpha:
    @dataclass
    class OpenOrder:
        prevOrderId: Optional[int] = None
        limitPrice: Optional[float] = None
        quantity: Optional[int] = None

    def __init__(self, signals: Dict[int, Signal], signal_ref: Dict[str, int], signal_col_ref: Dict[str, int] = None) -> None:
        self.signals = signals

        self.sig_to_row_map = signal_ref
        self.row_to_sig_map = { val: name for name, val in self.sig_to_row_map.items() }

        if signal_col_ref is None:
            self.name_to_col_map = {
                "TS": 0,
                "Open": 1,
                "High": 2,
                "Low": 3,
                "Close": 4,
                "Volume": 5,
                "BarCount": 6,
                "VWAP": 7,
            }
        else:
            self.name_to_col_map = signal_col_ref
        self.col_to_name_map = { val: name for name, val in self.name_to_col_map.items() }

    def call(self, signal_data):
        # high_price = (data1[-10:, 2] - data1[-10:, 7]).max()-0.25
        # low_price = (data1[-10:, 3] - data1[-10:, 7]).min()+0.25
        # weights = np.exp(-np.arange(9)/20)
        # trend = (np.diff(data1[-10:, 7], n=1) * weights).sum() / weights.sum() + data1[-1, 7]
        # high_price = round(4*(high_price + trend))/4
        # low_price = round(4*(low_price + trend))/4
        
        open_orders: Set[Alpha.OpenOrder] = set()
        return 






if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ClockController.set_utcnow(arrow.get(datetime.datetime(
        2020, 7, 15, 9, 30, 0), ClockController.time_zone))
    # robot_client = RobotClient(cliendId=0, live=False)
    robot_client = RobotClient(cliendId=0, simulator="influx")
    num_agents = 2
    es_key = robot_client.subscribe_asset('ES', 'GLOBEX', 'FUT', num_agents)
    # es_key = robot_client.subscribe_asset('SPY', 'SMART', 'STK')

    
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
    

    real_ts = arrow.utcnow()
    sim_ts = ClockController.utcnow()
    eod_ts = arrow.get(datetime.datetime(2020, 7, 15, 16, 00, 0), ClockController.time_zone)
    # eod_ts = ClockController.utcnow().replace(hour=16, minute=0, second=0)
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


        ClockController.increment_utcnow(1)
        time.sleep(0.025)
        # time.sleep(1.)
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


