# Chapter 2

It was brought to my attention in the [following issue](https://github.com/PacktPublishing/Python-for-Finance-Cookbook/issues/1) that there is a problem with loading the stock prices via the Yahoo Finance API within `backtrader`.

That is why I prepared a short tutorial [`pandas_datafeed_example.ipynb`] how to work around the issue by downloading the stock prices using `yfinance`, storing them as a `pandas` DataFrame and feeding it directly into `backtrader`.