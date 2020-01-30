from datetime import datetime
import backtrader as bt
from chapter_2_utils import MyBuySell

# create the strategy using a signal
class SmaSignal(bt.Signal):
    params = (('period', 20), )

    def __init__(self):
        self.lines.signal = self.data - bt.ind.SMA(period=self.p.period)

# download data
data = bt.feeds.YahooFinanceData(dataname='AAPL',
                                 fromdate=datetime(2018, 1, 1),
                                 todate=datetime(2018, 12, 31))

# create a Cerebro entity
cerebro = bt.Cerebro(stdstats = False)

# set up the backtest
cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.add_signal(bt.SIGNAL_LONG, SmaSignal)
cerebro.addobserver(MyBuySell)
cerebro.addobserver(bt.observers.Value)

# run backtest
print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
x = cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

# plot results
cerebro.plot(iplot=True, volume=False)
