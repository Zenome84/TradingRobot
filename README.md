# TradingRobot

Please get Interactive Brokers:
1. Download Trader Workstation (TWS): https://www.interactivebrokers.com/en/index.php?f=14099#tws-software
2. Open a paper trading account (I think you can do this from TWS)
3. Download the TWS_API: http://interactivebrokers.github.io/
4. Follow tutorials here to get started: https://tradersacademy.online/trading-courses/python-tws-api

The project currently has the following:
* HistDD.py - is a gui oriented example for downloading historical data
* ibapi_adapters - custom adapter to interface with TWS_API
* historical_data_cache - used to download historical tick data for futures and stocks
* robot_client.py - this is the main app that will comminate between the adapters and the eventual robot
  * You can currently use this to test 4 live streams on a single asset

Next steps:
* The work is divided into 3 parts
  * The Main Algo Classes
    * Signal Class: stores signals in a panda/numpy type
    * Asset Class: stores Signals
    * Order Manager Class: creates, updates (including cancel), and tracks orders
    * Alpha Class: these take Asset objects and transforms their Signals into new Signals
    * Other related classes: TBD
  * The TWS adapters
    * These will subscribe live streaming data using a handler from Main Algo
    * These will interface order from Order Manager
  * The simulation adapters
    * These will simulate live streaming data subscription from historical tick data and work the same way as equivalent TWS adapters
    * These will simulate orders using historical tick data
  * Machine Learning:
    * We will use the following work as a basis: http://www.cs.ucf.edu/~gitars/cap6671-2010/Presentations/Qlearning-stocks-multiagent.pdf
