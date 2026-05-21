
import time
import datetime
import pandas as pd

DATA = {}

def get_time() -> float:
    return datetime.datetime.now().isoformat()

def check_timer(label): 
    global DATA
    t = time.time()

    if label in DATA.keys(): 
        DATA[label] = t - DATA[label]
    else: 
        DATA[label] = t

def save_times(filename): 
    global DATA 
    pd.DataFrame(DATA, index=[0]).T.to_csv(filename)