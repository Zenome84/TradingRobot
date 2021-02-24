import time
import datetime
import arrow

import ibapi
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import ContractDetails

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mpl_dates

import mplfinance as mpf
import pandas as pd

from threading import Thread


def HA(df):
    df['HA_Close']=(df['Open']+ df['High']+ df['Low']+df['Close'])/4

    idx = df.index.name
    df.reset_index(inplace=True)

    for i in range(0, len(df)):
        if i == 0:
            df.at[i, 'HA_Open'] = ((df.at[i, 'Open'] + df.at[i, 'Close']) / 2)
        else:
            df.at[i, 'HA_Open'] = ((df.at[i - 1, 'HA_Open'] + df.at[i - 1, 'HA_Close']) / 2)

    if idx:
        df.set_index(idx, inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','High']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','Low']].min(axis=1)
    return df


# class msgHandler(object):
#     def handleHistoricalData(self, reqId: int, bar: BarData):
#         pass

#     def handleHistoricalDataEnd(self, reqId: int, start: str, end: str):
#         pass


class IBWrapper(EWrapper):
    def __init__(self):
        EWrapper.__init__(self)
        self.exporter = None
        self.contractDetailsIsObtained = False

    def init_withExporter(self, exporter):
        EWrapper.__init__(self)
        self.exporter = exporter
        self.contractDetailsIsObtained = False
        self.contractHistoricalDataIsObtained = False

    def historicalData(self, reqId: int, bar: BarData):
        """ returns the requested historical data bars

        reqId - the request's identifier
        date  - the bar's date and time (either as a yyyymmss hh:mm:ssformatted
             string or as system time according to the request)
        open  - the bar's open point
        high  - the bar's high point
        low   - the bar's low point
        close - the bar's closing point
        volume - the bar's traded volume if available
        count - the number of trades during the bar's timespan (only available
            for TRADES).
        WAP -   the bar's Weighted Average Price
        hasGaps  -indicates if the data has gaps or not. """

        if self.exporter == None:
            self.logAnswer(ibapi.utils.current_fn_name(), vars())
            return
        else:
            self.exporter.handleHistoricalData(reqId, bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        if self.exporter == None:
            self.logAnswer(ibapi.utils.current_fn_name(), vars())
        else:
            self.exporter.handleHistoricalDataEnd(reqId, start, end)

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        self.resolved_contract = contractDetails.contract

    def contractDetailsEnd(self, reqId: int):
        self.contractDetailsIsObtained = True


class IBClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


class tApp(IBWrapper, IBClient):
    def __init__(self, ipAddress, portId, clientId, msgHandler):
        IBWrapper.init_withExporter(self, msgHandler)
        IBClient.__init__(self, self)

        self.connect(ipAddress, portId, clientId)

        thread = Thread(target=self.run, name="MainThread")
        thread.start()

        setattr(self, "_thread", thread)


class Application(tk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        # msgHandler.__init__(self)

        self.port = 7496  # 7496: live; 7497: paper
        self.clientId = 0
        self.grid()
        self.create_widgets()
        self.isConnected = False

    def create_widgets(self):
        now = time.strftime('%Y%m%d %H:%M:%S',
                            time.localtime(int(time.time())))
        myfont = ('Arial', 12)
        varDateTimeEnd.set(now)

        self.btnConnect = ttk.Button(
            self, text="Connect", command=self.connect_to_tws)
        self.btnConnect.grid(row=0, column=1, sticky=tk.W)

        self.btnGetData = ttk.Button(
            self, text="Get Data", command=self.getHistoricalData)
        self.btnGetData.grid(row=0, column=2, sticky=tk.W)

        self.lable_datetime = tk.Label(
            root, font=myfont, text="End DateTime").grid(row=3, column=0)
        self.lable_duration = tk.Label(
            root, font=myfont, text="Duration").grid(row=3, column=1)
        self.lable_barsize = tk.Label(
            root, font=myfont, text="Bar Size").grid(row=3, column=2)

        self.cmbDateTimeEnd = tk.Entry(
            root, font=myfont, textvariable=varDateTimeEnd).grid(row=4, column=0)
        self.cmbDuration = ttk.Combobox(
            root, font=myfont, textvariable=varDuration)
        self.cmbDuration['values'] = ('1 Y', '6 M', '1 M', '7 D', '1 D')
        self.cmbDuration.grid(row=4, column=1)
        self.cmbBarSize = ttk.Combobox(root, font=myfont, textvariable=varBarSize)
        self.cmbBarSize['values'] = ('1 day', '1 hour', '20 mins', '10 mins', '5 mins', '2 mins', '1 min')
        self.cmbBarSize.grid(row=4, column=2)


        # self.listbox1 = tk.Listbox(root, font=("", 12), width=75, height=30)
        # self.listbox1.grid(row=6, column=0, columnspan=5,
        #                    padx=5, pady=5, sticky='w')

        # self.histDF = pd.DataFrame(
        #     columns=['Open', 'High', 'Low', 'Close', 'Volume']
        # )
        self.histCols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.fig1 = plt.Figure(figsize=(8, 6), dpi=100)
        plt.ion()

        ax1 = self.fig1.add_subplot(211)
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Price')
        ax2 = self.fig1.add_subplot(212)
        ax2.set_xlabel('DateTime')
        ax2.set_ylabel('Volume')

        self.chart1 = FigureCanvasTkAgg(self.fig1, root)
        self.chart1.get_tk_widget().grid(row=6, column=0, columnspan=5,
                         padx=5, pady=5, sticky='w')

        self.msgBox = tk.Listbox(root, font=("", 12), width=75, height=30)
        self.msgBox.grid(row=7, column=0, columnspan=5,
                         padx=5, pady=5, sticky='w')

    def connect_to_tws(self):
        if self.isConnected:
            self.tws_client.disconnect()
            self.btnConnect.config(text='Connect')
            self.msgBox.insert(tk.END, "Successfully disconnected")
            self.isConnected = False
        else:
            self.tws_client = tApp('127.0.0.1', self.port, self.clientId, self)
            timePassed = 0
            while not(self.tws_client.isConnected()):
                time.sleep(0.1)
                timePassed += 0.1
                if timePassed > 5:
                    self.msgBox.insert(
                        tk.END, "Waited more than 5 secs to establish connection")

            self.isConnected = True
            self.msgBox.insert(tk.END, "Successfully connected")
            self.btnConnect.config(text="Disconnect")

    def getHistoricalData(self):
        if not(self.isConnected):
            self.msgBox.insert(tk.END, "Not connected")
            return

        # self.listbox1.delete(0, tk.END)
        self.fig1.clf(keep_observers=True)
        ax1 = self.fig1.add_subplot(211)
        ax1.set_xlabel('DateTime')
        ax1.set_ylabel('Price')
        ax2 = self.fig1.add_subplot(212)
        ax2.set_xlabel('DateTime')
        ax2.set_ylabel('Volume')
        self.histData = {}

        self.contract = ibapi.contract.Contract()
        self.contract.symbol = "AAPL"
        self.contract.secType = "STK"
        self.contract.exchange = "SMART"
        self.contract.primaryExchange = "NASDAQ"
        self.contract.currency = "USD"

        self.tws_client.reqContractDetails(reqId=2, contract=self.contract)
        self.tws_client.contractDetailsIsObtained = False
        self.tws_client.contractHistoricalDataIsObtained = False

        timePassed = 0
        while not(self.tws_client.contractDetailsIsObtained):
            time.sleep(0.1)
            timePassed += 0.1
            if timePassed > 10:
                self.msgBox.insert(
                    tk.END, "Waited more than 10 secs for contract details")

        self.msgBox.insert(tk.END, "Successfully obtained contract details")
        aContract = self.tws_client.resolved_contract
        # aContract.includeExpired = True

        now = varDateTimeEnd.get()
        duration = varDuration.get()
        barsize = varBarSize.get()

        rth = 0
        self.tws_client.reqHistoricalData(reqId=1, contract=aContract, endDateTime=now,
                                          durationStr=duration, barSizeSetting=barsize,
                                          whatToShow='TRADES', useRTH=rth, formatDate=1,
                                          keepUpToDate=False, chartOptions=[])

        timePassed = 0
        while not(self.tws_client.contractHistoricalDataIsObtained):
            time.sleep(0.1)
            timePassed += 0.1
            if timePassed > 10:
                self.msgBox.insert(tk.END, "Waited more than 10 secs to download data")
                    
        df1 = pd.DataFrame.from_dict(self.histData, orient='index')
        df1.index.name = 'Date'
        df1.index = pd.to_datetime(df1.index)
        df1 = HA(df1)[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'Volume']]
        df1.columns = self.histCols
        mpf.plot(df1, type='candle', ax=ax1, volume=ax2, show_nontrading=bool(1-rth))
        ax1.xaxis.set_ticks(df1.index[::3])
        ax2.xaxis.set_ticks(df1.index[::3])
        
        self.fig1.tight_layout()
        self.fig1.canvas.draw()

    def disconnect(self):
        if self.isConnected:
            self.tws_client.disconnect()
            self.isConnected = False

    def handleHistoricalData(self, reqId: int, bar: BarData):
        if len(bar.date) <= 8:
            # date_fmt = '%Y%m%d'
            date_fmt = 'YYYYMMDD'
        else:
            # date_fmt = '%Y%m%d %H:%M:%S'
            date_fmt = 'YYYYMMDD  hh:mm:ss'
        # self.histData[datetime.datetime.strptime(bar.date, date_fmt)] = {
        self.histData[arrow.get(bar.date, date_fmt).shift(hours=-7).datetime] = {
            self.histCols[0]: bar.open,
            self.histCols[1]: bar.high,
            self.histCols[2]: bar.low,
            self.histCols[3]: bar.close,
            self.histCols[4]: bar.volume,
        }

    def handleHistoricalDataEnd(self, reqId: int, start: str, end: str):
        self.tws_client.contractHistoricalDataIsObtained = True
        self.msgBox.insert(tk.END, "Finished downloading data")


root = tk.Tk()

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        app.disconnect()
        root.destroy()

root.title("Historical Data from IB Python API")
root.geometry("800x1200")
root.attributes("-topmost", True)
root.protocol("WM_DELETE_WINDOW", on_closing)

varDateTimeEnd = tk.StringVar(root)
varDuration = tk.StringVar(root, value='1 D')
varBarSize = tk.StringVar(root, value='10 mins')

app = Application(root)

root.mainloop()
