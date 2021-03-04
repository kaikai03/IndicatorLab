
import pandas as pd
import copy

import QUANTAXIS as QA

from QAIndicatorStructExt import QA_DataStruct_Indicators_Ext

__LOW_FREQUENCE__ = [QA.FREQUENCE.YEAR, QA.FREQUENCE.QUARTER, QA.FREQUENCE.MONTH, QA.FREQUENCE.WEEK, QA.FREQUENCE.DAY]      


class Ind_Model:
    def __init__(self,data, ind_name, frequence=QA.FREQUENCE.DAY, pramas_default=None):
#         if not isinstance(data, type(QA.OUTPUT_FORMAT.DATASTRUCT)):
#             raise TypeError('Must be DATASTRUCT')
        if pramas_default is None:
            self._pramas_default = self.on_set_params_default()
        else:
            self._pramas_default = pramas_default
            
        if not isinstance(self._pramas_default, dict):
            raise TypeError('_pramas_default MUST BE DICT')
            
        self.pramas = copy.deepcopy(self._pramas_default)
        self.ind_name = ind_name
        self.data = data
        self.ind_df = None
        self.desition_df = None
        self.frequence = frequence
        

    def __repr__(self):
        return '< {} in pramas  {} ,{},{} >'.format(self.ind_name, 
                                                    str (self.pramas), 
                                                    'not fit' if not self.is_fitted else 'fitted:'+ str(self.keys_ind), 
                                                    'not desition' if not self.has_desition else 'has desition')
        
    def change_pramas(self,**dic):
        for k in dic.keys():
            self.pramas[k] = dic[k]
    
    def reset_pramas(self):
        self.pramas = copy.deepcopy(self._pramas_default)
    
    @property
    def cur_pramas(self):
        return self.pramas
    @property
    def default_pramas(self):
        return self._pramas_default
    @property
    def keys_pramas(self):
        return self._pramas_default.keys()
    @property
    def keys_ind(self):
        if not self.is_fitted:
            raise Exception("need fit the model first")
        return self.ind_df.keys().to_list()
    
    @property
    def is_fitted(self):
        return not self.ind_df is None
    
    @property
    def has_desition(self):
        return not self.desition_df is None
    
    @property
    def is_low_frequence(self):
        return self.frequence in __LOW_FREQUENCE__
    

    @property
    def ind(self):
        if not self.is_fitted:
            raise Exception("need fit the model first")
        return QA_DataStruct_Indicators_Ext(self.ind_df)

    @property
    def desition(self):
        if not self.has_desition  or not isinstance(self.desition_df, pd.DataFrame):
            print("desition_df:",self.desition_df)
            raise Exception('on_desition_structure error')
        return QA_DataStruct_Indicators_Ext(self.desition_df)
    

    def on_set_params_default(self) -> dict:
        raise NotImplementedError
        
    def on_indicator_structuring(self, data) -> pd.DataFrame:
        raise NotImplementedError
        
    def on_desition_structuring(self, data, ind_data) -> pd.DataFrame:
        raise NotImplementedError
        
        
        
    def excute_for_multicode(self, data, func, **pramas):
        return data.groupby(level=1, as_index=False, group_keys=False).apply(func,**pramas)

    def fit(self):
        self.ind_df = self.on_indicator_structuring(self.data)
        try:
            self.desition_df = self.on_desition_structuring(self.data, self.ind_df)
        except NotImplementedError:
            self.desition_df = None
        except Exception as err:
            raise Exception(err)
            
    