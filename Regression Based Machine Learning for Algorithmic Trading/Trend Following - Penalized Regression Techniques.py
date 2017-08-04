'''
Penalized Regression Techniques: Lasso, Ridge, ElasticNet

Penalized Regression Approach in Multi Asset Trend Following Strategy

We illustrate an application of Lasso by estimating the 1-day returns of assets in a cross-asset momentum model. We attempt to predict the returns of 4 assets: S&P 500, 7-10Y Treasury Bond Index, US dollar (DXY) and Gold. For predictor variables, we choose lagged 1M, 3M, 6M and 12M returns of these same 4 assets, yielding a total of 16 variables. 

Reference: JP Morgan Big Data and AI Strategies
'''

import pandas as pd
import numpy as np
from sklearn import linear_model

def initialize(context):

    #set_slippage(slippage.FixedSlippage(spread=0))
    #set_commission(commission.PerTrade(cost=1))
    set_symbol_lookup_date('2014-01-01')
    set_benchmark(symbol('IEF'))
    
    context.securities = symbol('IEF')
    context.reg_params = [symbol('SPY'), symbol('IEF'), symbol('UUP'), symbol('GLD')]

    context.inLong = False
    context.inShort = False
    
    schedule_function(my_rebalance, date_rules.month_start(days_offset=0), time_rules.market_open(hours=1))
     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())

 
def before_trading_start(context, data):
    pass

def my_record_vars(context,data):
    record(leverage=context.account.leverage)

def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    # dropping the last day to prevent look forward
    price_history = data.history(context.reg_params, fields='price',
                                 bar_count=750, frequency='1d')

    # Use returns history of the 4 tickers with different look back period as factor
    returns_mat_1m = price_history[:-1].pct_change(20).dropna()
    returns_mat_1m.columns = ['SPY_1m', 'IEF_1m', 'UUP_1m', 'GLD_1m']
    returns_mat_3m = price_history[:-1].pct_change(60).dropna()
    returns_mat_3m.columns = ['SPY_3m', 'IEF_3m', 'UUP_3m', 'GLD_3m']
    returns_mat_6m = price_history[:-1].pct_change(120).dropna()
    returns_mat_6m.columns = ['SPY_6m', 'IEF_6m', 'UUP_6m', 'GLD_6m']
    returns_mat_12m = price_history[:-1].pct_change(240).dropna()
    returns_mat_12m.columns = ['SPY_12m', 'IEF_12m', 'UUP_12m', 'GLD_12m']
    X = returns_mat_1m.join(returns_mat_3m).join(returns_mat_6m).join(returns_mat_12m).dropna()
    # X = X.dropna(axis=1)
   
    y = price_history[context.securities].pct_change().dropna()[-501:-1]
    
    # Transform by standardising the factors
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X[-501:-1])
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y)
    
    #reg = linear_model.LinearRegression(normalize=True)
    reg = linear_model.Lasso(alpha = 0.001, normalize=True)
    #reg = linear_model.Ridge (alpha = .5)
    reg.fit(X_train, y_train)
    
    res = reg.predict(sc_X.fit_transform(X[-501:-1])[-1])
    #record('predict', res, leverage=context.account.leverage)

    if get_open_orders():
        return

    # currently no position and predicted returns is positive, then open long
    if (res > 0) and (not context.inLong):
    # if (res > 0):    
        context.inLong = True
        context.inShort = False
        order_target_percent(context.securities, 1.0)
        # print('scenario 3', context.inShort, context.inLong)
        return

    # currently no position and predicted returns is negative, then open short
    if (res < 0) and (not context.inShort):
    # if (res < 0):    
        context.inLong = False
        context.inShort = True
        order_target_percent(context.securities, -1.0)
        # print('scenario 4', context.inShort, context.inLong)      
        return

        
    
    
