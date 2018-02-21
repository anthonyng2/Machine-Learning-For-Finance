# Machine Learning For Finance

# 1. Regression Based Machine Learning for Algorithmic Trading

[Machine Learning for Finance, Algorithmic Trading and Investing Slides](https://github.com/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Machine%20Learning%20-%20Linear%20Regression%20for%20Algo%20Trading%20v2017-07-13.pdf)

These set of slides explained the current asset management environment and the advanced of technology on asset management. Categories of Machine and Deep Learning are explained. A brief introduction on linear regression and associated assumptions are covered. Stylized statistical properties of financial time series and asset returns are presented highlighting the challenges. 

To ease learners to understand machine learning, linear regression has been used as the conduit. Firstly, the shortcoming of linear regression is highlighted. We then follow by the steps of model building and covering concepts such as hyperparameters, cross-validation, model validation, bias-variance tradeoff. The 6 stages of professional quant strategy is also covered to provide some perspective on where machine learning fits in.



# 1.1 Pairs Trading & Machine Learning

## Linear Regression
[A Walk Through on How to Design Your Own Pairs Trading Using Linear Model](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs_Trading_and_Linear_Regression.ipynb)

[Notebook - Introduction to Linear Regression and Machine Learning Model Building Process](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Linear%20Regression.ipynb)

## Moving to Backtesting

### Statsmodel - Linear Regression
[Quantopian IDE codes for Pairs Trading using Linear Regression Model - statsmodel Pre 2008](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs%20Trading%20statsmodels%20Linear%20Pre%202008.py) and [Quantopian IDE codes for Pairs Trading using Linear Regression Model - statsmodel Post 2008](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs%20Trading%20statsmodels%20Linear%20Post%202008.py)

This backtest utilise Python statsmodel to build the linear regression model. We then move on to illustrate how one can use the Python scikit-learn model to do likewise.

### scikit-learn - Linear Regression
[Quantopian IDE codes for Pairs Trading using Linear Regression Model - scikit-learn](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs%20Trading%20scikit-learn%20Linear.py)

### scikit-learn - Lasso Regression
[Lasso Regression](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs%20Trading%20-%20Lasso%20Regression.py)

### scikit-learn - Ridge Regression
[Ridge Regression](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs%20Trading%20-%20Ridge%20Regression.py)

### scikit-learn - Bayesian Ridge Regression
[Bayesian Ridge Regression](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs%20Trading%20-%20Bayesian%20Ridge%20Regression.py)

### scikit-learn - ElasticNet Regression
[ElasticNet Regression](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs%20Trading%20-%20Elastic%20Net.py)



# 1.2 Pairs Trading and Kalman Filter
[Pairs Trading Design with Kalman Filter](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Pairs_Trading_with_Linear_Regression_and_Kalman_Filter.ipynb)



# 1.3 Trend Following & Machine Learning

[Trend Following Strategies with Machine Learning](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Trend_Following_Strategies_Penalized_Regression_Approach.ipynb)



# 1.4 References:
[UCL - Characterization of Financial Time Series](https://github.com/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/RN_11_01.pdf)

[Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues. Rama Cont](https://github.com/anthonyng2/Machine-Learning-For-Finance/blob/master/Regression%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/empirical%20properties%20of%20asset%20returns.pdf)





# 2. Classification Based Machine Learning for Algorithmic Trading

This portion is under active development at the moment. I have uploaded some of my codes and backtesting results. A common use of classification ML is to predict the next day's result. You can find some examples [here](https://www.quantstart.com/articles/Forecasting-Financial-Time-Series-Part-1). Some of the ML classification methods were capable of achieving prediction accuracy of pver 60%. Does that translate directly to returns and out-performance over simple buy-and-hold strategy? Check out the backtesting tearsheets for the answer. 

[Classification Based Machine Learning Algorithm](https://nbviewer.jupyter.org/github/anthonyng2/Machine-Learning-For-Finance/blob/master/Classification%20Based%20Machine%20Learning%20for%20Algorithmic%20Trading/Classification%20Based%20Machine%20Learning%20Algorithm.ipynb)








