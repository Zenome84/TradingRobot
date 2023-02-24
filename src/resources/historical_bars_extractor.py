
import json
import time
import arrow
import datetime
import numpy as np
from tqdm import tqdm
from core.robot_client import RobotClient
from models.asset import Asset
from resources.enums import BarSize
from resources.time_tools import ClockController, wait_until


if __name__ == "__main__":
    robot_client = RobotClient(cliendId=0, simulator="influx")
    
    valid_days = Asset.getValidTradingDays('ES', 'GLOBEX', robot_client,
        arrow.get(datetime.datetime(2018, 2, 1, 0, 0, 0), ClockController.time_zone),
        arrow.get(datetime.datetime(2022, 2, 1, 0, 0, 0), ClockController.time_zone),
        minValidActivity=500000, numLookbackDays=2
    )
    allData = dict()

    for valid_day in tqdm(valid_days):
        bod_ts = arrow.get(datetime.datetime(
            valid_day.year, valid_day.month, valid_day.day, 9, 30, 0), ClockController.time_zone)
        eod_ts = arrow.get(datetime.datetime(
            valid_day.year, valid_day.month, valid_day.day, 16, 00, 0), ClockController.time_zone)
        
        ClockController.set_utcnow(bod_ts)
        
        es_key = robot_client.subscribe_asset('ES', 'GLOBEX', 'FUT')
        
        wait_until(
            condition_function=lambda: robot_client.assetCache[es_key].updateOrderRulesObtained,
            seconds_to_wait=5,
            msg=f"Waited more than 5 secs to update order rules."
        )

        reqId1 = robot_client.subscribe_bar_signal(es_key, BarSize.SEC_01, 600)
        reqId2 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_01, 100)
        reqId3 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_05, 100)
        reqId4 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_15, 100)
        reqId5 = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_30, 100)
        # reqId6 = robot_client.subscribe_bar_signal(es_key, BarSize.HRS_01, 50)

        prev_sec_elapsed: int = 0
        sec_idx_value: int = 0
        sec01data: np.ndarray = np.ndarray([0,8], float)
        min01data: np.ndarray = np.ndarray([0,8], float)
        min05data: np.ndarray = np.ndarray([0,8], float)
        min15data: np.ndarray = np.ndarray([0,8], float)
        min30data: np.ndarray = np.ndarray([0,8], float)
        # hour1data = np.ndarray([0,8], float)
        while ClockController.utcnow() <= eod_ts:
            sec_elapsed = (ClockController.utcnow() - bod_ts).seconds

            if sec_elapsed % 600 == 0: # 10 min
                data = robot_client.assetCache[es_key].signals[reqId1].get_numpy()
                whr = np.where(data[:, 0] == sec_idx_value)[0]
                if len(whr) == 0:
                    sec_idx = 0
                else:
                    sec_idx = whr[0] + 1
                sec_idx_value = data[-1, 0]
                sec01data = np.append(sec01data, data[sec_idx:], 0)
            if sec_elapsed % 1800 == 0: # 30 min
                min01data = np.append(min01data, robot_client.assetCache[es_key].signals[reqId2].get_numpy()[-(sec_elapsed-prev_sec_elapsed)//60:], 0)
                min05data = np.append(min05data, robot_client.assetCache[es_key].signals[reqId3].get_numpy()[-(sec_elapsed-prev_sec_elapsed)//60//5:], 0)
                min15data = np.append(min15data, robot_client.assetCache[es_key].signals[reqId4].get_numpy()[-(sec_elapsed-prev_sec_elapsed)//60//15:], 0)
                min30data = np.append(min30data, robot_client.assetCache[es_key].signals[reqId5].get_numpy()[-(sec_elapsed-prev_sec_elapsed)//60//30:], 0)
                # hour1data = np.append(hour1data, robot_client.assetCache[es_key].signals[reqId6].get_numpy()[-(sec_elapsed-prev_sec_elapsed)//60//60:], 0)
                prev_sec_elapsed = sec_elapsed

            # if ClockController.utcnow().shift(seconds=600) <= eod_ts:
            ClockController.increment_utcnow(600)
            # else:
            #     ClockController.increment_utcnow(60)
            time.sleep(1)
        
        robot_client.unsubscribe_asset('ES', 'GLOBEX')
        time.sleep(1)
        allData[str(valid_day.int_timestamp)] = {
            BarSize.SEC_01.value: sec01data.tolist(),
            BarSize.MIN_01.value: min01data.tolist(),
            BarSize.MIN_05.value: min05data.tolist(),
            BarSize.MIN_15.value: min15data.tolist(),
            BarSize.MIN_30.value: min30data.tolist()
        }
    
    with open('Py/allData.json', 'w') as fwrite:
        json.dump(allData, fwrite)
    
    # with open('Py/allDataTemp.json', 'r') as fread:
    #     allDataLoad = json.load(fread)

    exit()
