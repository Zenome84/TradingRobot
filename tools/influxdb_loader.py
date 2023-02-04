from datetime import datetime
import arrow
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mplf

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from resources.time_tools import ClockController

# You can generate a Token from the "Tokens Tab" in the UI
token = "_W_lIVGTgVIfmloET33KC95vL9Qzx3hdIkePQTGNv5hlaGpLn-Oy1ndGN4LhEflBNoKSM1D3eddRXO_rY-FguA=="
org = "Zenome"
bucket = "MarketData"
symbol = "ES"
exchange = "GLOBEX"

client = InfluxDBClient(url="http://localhost:8086", token=token)

write_api = client.write_api(write_options=SYNCHRONOUS)

sequence = []
cache_key = f"{symbol}_{exchange}"
cache_path = f"U:/MarketData/{cache_key}"
cache_files = glob.glob(cache_path + "/*")
ts_prev = 0
tick_idx = 0
rows = 0
for contract_path in cache_files:
    contract_files = glob.glob(contract_path + "/*.json")
    contract = int(contract_path[-8:])
    for contract_file in contract_files:
        with open(contract_file, 'r') as cf:
            tick_data = json.load(cf)
            tick_data = dict(sorted(tick_data.items()))
            for key, tick in tick_data.items():
                ts_curr = tick['time_ts']
                if ts_prev == ts_curr:
                    tick_idx += 1
                else:
                    tick_idx = 0
                ts_prev = ts_curr
                sequence.append(
                    f"tick,"
                    + f"exchange={exchange},symbol={symbol},contract={contract} "
                    + f"price={tick['price']:0.2f},size={tick['size']},ps={tick['price']*tick['size']:0.2f} "
                    + f"{ts_curr*10**9+tick_idx}"
                )
                if len(sequence) > 100:
                    write_api.write(bucket, org, sequence)
                    rows += len(sequence)
                    sequence = []
                    print(f"Loaded {rows:10d} rows")

write_api.write(bucket, org, sequence)
rows += len(sequence)
sequence = []
print(f"Loaded {rows:10d} rows")


# with open('tools/influxdb_query.influx', 'r') as file:
#     query = file.read()

#     # .replace("v.windowPeriod", "1d, offset: 4h") \
# query = query.replace("v.timeRangeStart", str(arrow.get("2019-06-20 11:40:00.000-04:00").int_timestamp)) \
#     .replace("v.timeRangeStop", str(arrow.get("2019-06-20 12:45:00.000-04:00").int_timestamp)) \
#     .replace("v.windowPeriod", "30s") \
#     .replace("v.exchange", "GLOBEX") \
#     .replace("v.symbol", "ES") \
#     .replace("v.contract", "20190621") \

# tables = client.query_api().query(query, org=org)

# print("  TS                      |  Open    |  High    |  Low     |  Close   |  VWAP    |  Volume  |  Trades  |")
# print("--------------------------------------------------------------------------------------------------------")
# histData = {}
# histCols = ['Open', 'High', 'Low', 'Close', 'Volume']
# for record in tables[0].records:
#     row = record.values
#     print(f"{arrow.get(row['_time']).to(ClockController.time_zone)} |{row['open']:9.2f} |{row['high']:9.2f} |{row['low']:9.2f} |{row['close']:9.2f} |{row['vwap']:9.2f} |{row['volume']:9.0f} |{row['count']:9.0f} |")
#     histData[arrow.get(row['_time']).to(ClockController.time_zone).datetime] = {
#             histCols[0]: row['open'],
#             histCols[1]: row['high'],
#             histCols[2]: row['low'],
#             histCols[3]: row['close'],
#             histCols[4]: row['volume'],
#         }

# fig = plt.figure(figsize=(8, 6), dpi=100)
# # plt.ion()

# ax1 = fig.add_subplot(211)
# ax1.set_xlabel('DateTime')
# ax1.set_ylabel('Price')
# ax2 = fig.add_subplot(212)
# ax2.set_xlabel('DateTime')
# ax2.set_ylabel('Volume')


# def HA(df):
#     df['HA_Close']=(df['Open']+ df['High']+ df['Low']+df['Close'])/4

#     idx = df.index.name
#     df.reset_index(inplace=True)

#     for i in range(0, len(df)):
#         if i == 0:
#             df.at[i, 'HA_Open'] = ((df.at[i, 'Open'] + df.at[i, 'Close']) / 2)
#         else:
#             df.at[i, 'HA_Open'] = ((df.at[i - 1, 'HA_Open'] + df.at[i - 1, 'HA_Close']) / 2)

#     if idx:
#         df.set_index(idx, inplace=True)

#     df['HA_High']=df[['HA_Open','HA_Close','High']].max(axis=1)
#     df['HA_Low']=df[['HA_Open','HA_Close','Low']].min(axis=1)
#     return df

# df1 = pd.DataFrame.from_dict(histData, orient='index')
# df1.index.name = 'Date'
# df1.index = pd.to_datetime(df1.index)
# df1 = HA(df1)[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'Volume']]
# df1.columns = histCols
# mplf.plot(df1, type='candle', ax=ax1, volume=ax2, show_nontrading=True)
# # ax1.xaxis.set_ticks(df1.index[::3])
# # ax2.xaxis.set_ticks(df1.index[::3])

# fig.tight_layout()
# print(f"Done querying {len(tables[0].records)} rows")
# plt.show()

# print(query)