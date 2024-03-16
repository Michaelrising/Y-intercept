import pandas as pd
import numpy as np


class FactorMaker:
    def __init__(self, data):
        self.data = data
        self.data.set_index('date', inplace=True)

    def make_factor(self):
        factors = []
        for ticker, data in self.data.groupby('ticker'):
            factor = self._make_factor(data.sort_index())
            factor['ticker'] = ticker
            factors.append(factor)
        factors = pd.concat(factors).reset_index()
        return factors

    def _make_factor(self, data):
        factor = pd.DataFrame(index=data.index)
        factor = self.vwap(factor, data)
        factor = self.mom(factor, data)
        factor = self.sma(factor, data)
        factor = self.rsi(factor, data)
        factor = self.macd(factor, data)
        factor = self.obv(factor, data)
        factor = self.reg_factors(factor, data)
        return factor

    def vwap(self, factor, data, time_period=10):
        factor['vwap'] = (data['last'] * data['volume']).rolling(time_period, min_periods=2).sum() / data['volume'].rolling(time_period, min_periods=2).sum()
        return factor

    def mom(self, factor, data, time_period=10):
        factor['mom'] = (data['last'] - data['last'].shift(time_period))/data['last'].shift(time_period)
        return factor

    def sma(self, factor, data, time_period=10):
        factor['sma'] = data['last'].rolling(time_period).mean()
        return factor

    def rsi(self, factor, data, time_period=14):
        delta = data['last'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=time_period).mean()
        avg_loss = loss.rolling(window=time_period).mean()
        rs = avg_gain / avg_loss
        factor['rsi'] = 100 - (100 / (1 + rs))
        return factor

    def macd(self, factor, data, short_period=12, long_period=26, signal_period=9):
        short_ema = data['last'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['last'].ewm(span=long_period, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        factor['macd'] = macd_line
        factor['signal'] = signal_line
        return factor

    def obv(self, factor, data):
        price_changes = data['last'].diff()
        volume = np.log(data['volume'] + 10e-7)
        obv = pd.Series(0.0, index=data.index)
        for loc, i in enumerate(price_changes.index[1:]):
            pre_i = price_changes.index[loc-1]
            if price_changes[i] > 0:
                obv[i] = obv[pre_i] + volume[i]
            elif price_changes[i] < 0:
                obv[i] = obv[pre_i] - volume[i]
            else:
                obv[i] = obv[pre_i]
        factor['obv'] = obv.values
        return factor

    def reg_factors(self, factor, data, timeperiod=14, nbdev=1):
        reg = data['last'].rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        reg_angle = np.arctan(reg) * (180 / np.pi)
        reg_intercept = data['last'].rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[1], raw=True)
        reg_slope = data['last'].rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        stddev = data['last'].rolling(window=timeperiod).std() * nbdev
        tsf = data['last'].rolling(window=timeperiod).mean()
        var = data['last'].rolling(window=timeperiod).var() * nbdev
        factor['linear_reg'] = reg
        factor['linear_reg_angle'] = reg_angle
        factor['intercept'] = reg_intercept
        factor['slope'] = reg_slope
        factor['stddev'] = stddev
        factor['tsf'] = tsf
        factor['var'] = var
        return factor


