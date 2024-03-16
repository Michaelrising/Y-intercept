# Y-intercept Coding Test 
### 2024-03-16 | LU YITAO

## 1. Introduction
This is th Y-intercept coding test, which I was required to generate a simple trading strategy by python and backtest it.
The market data provided included daily close price and traded volume, and the other two files including the daily market
capital and the sector information. 

Since the data provided is not quite informative to derivative many features for model building, I first try a quite simple
strategy called Grids Trading strategy. But this strategy does not explore all the data provided, and the performance in four
years is OK but not quite enough. So I then try using the data as much as I can and also derive some new features to build
a predictive model. And then using the prediction of the model to generate trading signals and backtest the strategy.

In the strategy, we assume that we could trade at the close price once we see it and we could trade at any volume as we want. 

## 2. Data Preprocessing
There are two parts for data processing, the first part is to process the close data for the grids trading strategy, and the
second part, which is more complex, is to process the data for the model building.

## 3 Grids Trading Strategy
several steps to be followed:
    
    1) Calculate the volatility of the close price by ewm_std(), here the spanning window is a hyper parameter 
    to be fine-tuned.
    2) Calculate the Upper and Lower bounds for the grids by rolling max and min of the close price. The grids
    are equally spaced to 10 parts between the upper and lower bounds.
    3) Generate the trading signals based on the close price and the grids: if the close price gose up and touch
    the grid, we short 100 shares, else if the close price goes down and touch the grid, we long 100 shares.
    4) In every Mondy we reset the grids for each stock. 

Indeed, this strategy is quite simple and as you can see the performance for a default setting has annualized return of 
about 10% and the Sharpe ratio is about 1.43 and the maximum drawdown is about 10%. The detailed performance is shown in the
results folder: results/backtest_easy_run_2. 

There are several parameters could be fine-tuned for the strategy, including the window for the volatility, the window for 
calculating the grids, the number of grids, the amount of shares to trade, and the grids-reset frequency.

## 4 LGBM Model Strategy
To build the model and train the model, we have to do many feature engineering and data preprocessing, including deriving
factors from the close price, the volume, the market capital and the sector information and the label Y. 
Since we only have the close price of the ohlc price, many technical indicators are impossible to derive. Hence, the indicators 
are defined in utils/factors.py. 

However, we find out the results for lgbm model is ot better than the grids trading strategy, with an annualized return of 0.085 and
maximum drawdown of 0.23 and the sharpe ratio is 1.23. Results are saved in:  results/backtest_lgbm_run_3. 

This shows that with insufficient data, which we cannot generate enough features for the model building, the model is not
performing well. But we can still fine tune the model parameters and the features to improve the performance.

