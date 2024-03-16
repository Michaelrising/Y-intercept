import pandas as pd

def SMA(values, n):
    """
    Return Simple Moving Average
    """
    return pd.Series(values).rolling(n).mean()


def assert_msg(condition, msg):
    if not condition:
        raise Exception(msg)

