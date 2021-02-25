# TradingRobot

Please get Interactive Brokers:
1. Download Trader Workstation: https://www.interactivebrokers.com/en/index.php?f=14099#tws-software
2. Download the TWS_API: http://interactivebrokers.github.io/
3. Follow tutorials here to get started: https://tradersacademy.online/trading-courses/python-tws-api

The project currently has the following:
* HistDD.py - is a gui oriented example for downloading historical data
* ibapi_adapters - custom adapter to interface with TWS_API
* historical_data_cache - used to download historical tick data for futures and stocks

Next steps:
* The work is divided into 3 parts
  * The Main Algo Classes
    * Signal Class: stores signals in a panda/numpy type
    * Asset Class: stores Signals
    * Other related classes: TBD
  * The TWS adapters
    * These will subscribe live streaming data using a handler from Main Algo
  * The simulation adapters
    * These will simulate live streaming data subscription from tick data and work the same way as equivalent TWS adapters
