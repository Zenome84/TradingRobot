from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
1
if __name__ == '__main__':
    # cerebro = bt.Cerebro()
    # cerebro.broker.setcash(100000.0)

    # print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # cerebro.run()

    # print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    ibstore = bt.stores.IBStore(host='127.0.0.1', port=7497, clientId=35)
    data = ibstore.getdata(dataname='ES-YYYYMM-GLOBEX')