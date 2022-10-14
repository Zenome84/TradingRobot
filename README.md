# TradingRobot

Please get Interactive Brokers:
1. Download Trader Workstation (TWS): https://www.interactivebrokers.com/en/index.php?f=14099#tws-software
2. Open a paper trading account (I think you can do this from TWS)
3. Download the TWS_API: http://interactivebrokers.github.io/
4. Follow tutorials here to get started: https://tradersacademy.online/trading-courses/python-tws-api
5. Get influxdb and set it up: https://portal.influxdata.com/downloads/

The project currently has the following:
* `HistDD.py` - is a gui oriented example for downloading historical data and is a good way to visualize data
* `ibapi_adapter.py` - custom adapter to interface with TWS_API
  * Supports requesting bar/tick data + historical and placing/cancelling orders (modifying orders seems to not work)
* `sim_adapter.py` - custom adapter to interface with influx_db
  * Supports requesting bar data and placing/cancelling limit orders for *futures*
* `historical_data_cache.py` - used to download historical tick data for *futures* and *stocks*
* `influxdb_loader.py` - takes output from `historical_data_cache` and puts it to an influxdb configured to localhost
* `robot_client.py` - framework with which to communicate through either the simmulated, database, or live environments
  * Supports requesting multiple bar data signals for *futures* and *stocks*
  * Supports multiple-agent limit order management on *futures* with PnL tracking by agent

Next steps:
* The work is divided into 3 parts
  * The Main Algo Classes
    * **[Done]** Signal Class: stores signals in a panda/numpy type
    * **[Done]** Asset Class: stores Signals
    * **[Done]** Order Manager Class: creates, updates (including cancel), and tracks orders
      * Not as a separate class anymore, but embedded into `robot_client.py`
    * Alpha Class: these take Asset objects and transforms their Signals into new Signals
    * Other related classes: TBD
  * The TWS adapters
    * **[Done]** These will subscribe live streaming data using a handler from Main Algo
    * **[Done]** These will interface order from Order Manager
      * Simply interfacing through `robot_client.py` 
  * The simulation adapters
    * **[Done]** These will simulate live streaming data subscription from historical tick data and work the same way as equivalent TWS adapters
    * **[Done]** These will simulate orders using historical tick data queried from influxdb
  * Machine Learning:
    * We will use the following work as a basis: http://www.cs.ucf.edu/~gitars/cap6671-2010/Presentations/Qlearning-stocks-multiagent.pdf
