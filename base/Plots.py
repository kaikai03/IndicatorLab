
import numpy as np
import mpl_finance as mpf
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num, num2date
import matplotlib.ticker as ticker

class base_formatter(ticker.Formatter):
    ''' 不在图表上直接使用时间。
        对位将xaxis换成对应的时间。
    '''
    def __init__(self, dates, fre_type='day'):
        self.dates = dates
        if 'min' in fre_type:
            self.fmt = '%m%d %H:%M'
        else:
            self.fmt = '%m%d'
    
    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        if x < 0 :
            return '-'
        dates_len = len(self.dates)
        if x >= dates_len:
            if dates_len < 2:
                return '-'
            return (self.dates[dates_len-1]+(self.dates[1]-self.dates[0])*(x-dates_len)).strftime(self.fmt)
        
        ind = int(np.round(x))
        #ind就是x轴的刻度数值，不是日期的下标
        return self.dates[ind].strftime(self.fmt)
    
def base_plot(code, stock_struct):
    date_descript = 'date'
    if 'min' in stock_struct.type :
        date_descript = 'datetime'
        
    fig,(ax1,ax2)=plt.subplots(2,sharex=True,figsize=(1120/72,420/72),
                               gridspec_kw={
                                   'width_ratios': [1],
                                   'height_ratios': [3, 1]})
    data_code = stock_struct.data.loc[(slice(None), code), :]
    a = data_code.reset_index()[[date_descript,'open','close','high','low','volume']]

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(10))
    formatter = base_formatter(a[date_descript],stock_struct.type)
    ax1.xaxis.set_major_formatter(formatter)

    p = mpf.candlestick2_ohlc( ax1, a['open'].values,a['high'].values,a['low'].values,a['close'].values, 
                              colordown='#53c156', colorup='#ff1717',width=0.7,alpha=1)

    grey_pred = np.where(a["close"] == a["open"], a['volume'], 0)
    red_pred = np.where(a["close"] > a["open"], a['volume'], 0)
    green_pred = np.where(a["close"] < a["open"], a['volume'], 0)

    ax2.bar(list(np.arange(len(a[date_descript]))), grey_pred, facecolor="grey",width=0.7)
    ax2.bar(list(np.arange(len(a[date_descript]))), red_pred, facecolor="#ff1717",width=0.7)
    ax2.bar(list(np.arange(len(a[date_descript]))), green_pred, facecolor="#53c156",width=0.7)

    ax2.set_ylabel('Volume')
    ax2.autoscale_view()