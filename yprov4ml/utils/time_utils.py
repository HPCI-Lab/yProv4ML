
import time
import datetime
import pandas as pd

DATA = {}

def get_time() -> float:
    return datetime.datetime.now().isoformat()

def timestamp_to_minutes(ts):
    return ts / 60000

def timestamp_to_seconds(ts):
    return ts / 1000


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