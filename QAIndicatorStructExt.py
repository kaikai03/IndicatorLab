
from QUANTAXIS.QAData import QA_DataStruct_Indicators
    
class QA_DataStruct_Indicators_Ext(QA_DataStruct_Indicators):
    def __init__(self, data):
        self.data = data
        super().__init__(data)
        
    def get_ind(self, code=None, date=None):
        if code and date :
            if isinstance(date,tuple):
                return self.data.loc[pd.IndexSlice[date[0]:date[1],code],:]
            return self.data.loc[pd.IndexSlice[date,code],:]
        if code :
            return self.data.xs(code, level=1)
        if date :
            if isinstance(date,tuple):
                return self.data.loc[date[0]:date[1]] 
            return self.data.loc[date]
            
        raise Exception("不允许两个参数都为空")
        
    def get_ind_offset(self, code, base_date, offset=0):
        if code is None or base_date is None:
            raise Exception("code，date 不允许为空")
            
        ind = self.data.xs(code, level=1)
        idx = ind.index.get_loc(base_date)
        
        if np.sign(offset) < 0 :
            return ind.iloc[idx + offset if idx + offset >=0 else 0:idx+1]
        else:
            return ind.iloc[idx:idx+offset+1]

        
    