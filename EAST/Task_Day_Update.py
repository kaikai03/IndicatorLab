
import os
import sys
import time

module_path = os.path.abspath(os.path.join('..')) 
if module_path not in sys.path: 
    sys.path.append(module_path)
    
from base.JuUnits import now_time_tradedate

from QUANTAXIS.QAUtil import (
    DATABASE)
from QUANTAXIS.QAUtil.QADate_trade import (
    QA_util_if_trade)

from Crawler_Block_East import (
    update_east_stock_block,
    update_all_block_kline,
    update_1min_block_kline)

from Crawler_North_East import (
    update_north_line,
    update_north_1min,
    update_deal_top10)




stdout_tmp = None
__log_file_path__ = './log_file/'
__log_postfix__ = '.log'

def start_log():
    global stdout_tmp
    stdout_tmp = sys.stdout
    sys.stdout = Logger()

def stop_log():
    global stdout_tmp
    if not stdout_tmp is None:
        sys.stdout = stdout_tmp
        stdout_tmp = None
    
def get_cur_time():
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    
def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
class Logger(object):
    def __init__(self):
        create_dir_not_exist(__log_file_path__)
        self.terminal = sys.stdout
        self.date_str = get_cur_time()[0:10]
        self.log = open(__log_file_path__+self.date_str+__log_postfix__, "a")
        

    def write(self, message):
        ##防止正好隔日
        if self.date_str != get_cur_time()[0:10]:
            self.log.close()
            self.date_str = get_cur_time()[0:10]
            self.log = open(__log_file_path__ + self.date_str + __log_postfix__, "a")

        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()
        
def fetch_update_date():
    coll = DATABASE.update_em
    res = coll.find_one({})
    if not res is None:
        res = res['datetime']
    return res
    
def update_update_date():
    coll = DATABASE.update_em
    coll.delete_one({})
    coll.insert_one({'datetime':get_cur_time()})

def is_today_updated():
    """今日是否已经更新过,更新过了返回True
    """
    t = fetch_update_date()
    if t is None:
        return False
    return fetch_update_date()[0:10] == get_cur_time()[0:10]

def main():
    start_log()
    if is_today_updated():
        print('今日已完成过更新')
        stop_log()
        return
    
    today = get_cur_time()[0:10]
    
    if int(get_cur_time()[8:10]) % 4 == 0:
        update_east_stock_block() # 更新block
        
    if QA_util_if_trade(today):
        update_north_line(mode='fast') # 更新北向日线数据
        time.sleep(10)
        update_north_1min() # 更新北向1min数据
        time.sleep(10)
        update_all_block_kline(verbose=False) # 更新k线
        time.sleep(30)
        update_1min_block_kline(verbose=False) # 获取今日 1min k线
        time.sleep(30)
        update_deal_top10(verbose=True) # 更新北向交易top10数据
        time.sleep(5)
        
    

    print('更新完成')
    update_update_date()
    stop_log()
    
def init():
    start_log()

    update_east_stock_block() # 更新block
    
    update_north_line(mode='init') # 更新北向日线数据
    time.sleep(10)
    update_north_1min() # 更新北向1min数据
    time.sleep(10)
    update_all_block_kline(verbose=True) # 更新k线
    time.sleep(30)
    update_1min_block_kline(verbose=True) # 获取今日 1min k线
    time.sleep(30)
    update_deal_top10(verbose=True) # 更新北向交易top10数据
    time.sleep(5)
        

    print('更新完成')
    update_update_date()
    stop_log()

if __name__ == '__main__':
    main()
     # init()
