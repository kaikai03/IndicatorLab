
from IPython.display import display
import inspect

display_handle = None
def stream_print_on():
    global display_handle
    display_handle = display("display",display_id=True)

def stream_print(str):
    assert display_handle!=None,"需要先stream_print_on,或使用装饰器"
    display_handle.update(str)

def stream_print_off():
    global display_handle
    display_handle = None
    
def stream_print_wrap(f):
    def inner(*args,**kwargs):
        stream_print_on()
        ret =f(*args,**kwargs)
        stream_print_off()
        return ret
    return inner

def help_source_code(moulde):
    return inspect.getsourcelines(moulde)

def excute_for_multidates(data, func, level=0, **pramas):
    return data.groupby(level=level, group_keys=False).apply(func,**pramas)