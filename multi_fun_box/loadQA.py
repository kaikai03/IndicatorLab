def loadQA(x):
    import time
    starttime = time.time()
#     from QUANTAXIS.QAFetch.QAQuery_Advance import QA_fetch_stock_day_adv
    import QUANTAXIS as QA
    end1 = time.time() - starttime
    
    starttime = time.time()
    data = QA.QA_fetch_stock_day_adv(x, '2005-05-29', '2020-06-29')
    end2 = time.time() - starttime
#     return (x,end1,end2)
    return data.data
