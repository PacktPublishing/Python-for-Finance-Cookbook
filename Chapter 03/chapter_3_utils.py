import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
warnings.simplefilter(action='ignore', category=InterpolationWarning)
plt.style.use('seaborn')

def adf_test(series):
    '''
    Function for running the Augmented Dickey Fuller test and displaying the results in a human-readable format. The number of lags is chosen automatically based on AIC.
    
    Null Hypothesis: time series is not stationary
    Alternate Hypothesis: time series is stationary

    Args:
    * series - a pd.Series object containing the time series to investigate
    '''
    adf_test = adfuller(series, autolag='AIC')
    adf_results = pd.Series(adf_test[0:4], index=['Test Statistic',
                                                  'p-value',
                                                  '# of Lags Used',
                                                  '# of Observations Used'])
    for key, value in adf_test[4].items():
        adf_results[f'Critical Value ({key})'] = value

    print('Results of Augmented Dickey-Fuller Test ----')
    print(adf_results)


def kpss_test(series, h0_type='c'):
    '''
    Function for running the KPSS test and displaying the results in a human-readable format.

    Null Hypothesis: time series is stationary
    Alternate Hypothesis: time series is not stationary

    When null='c' then we are testing level stationary, when 'ct' trend stationary.

    Args:
    * series - a pd.Series object containing the time series to investigate
    * h0_type - string, what kind of null hypothesis is tested
    '''
    kpss_test = kpss(series, regression=h0_type)
    kpss_results = pd.Series(kpss_test[0:3], index=['Test Statistic',
                                                    'p-value',
                                                    '# of Lags'])
    for key, value in kpss_test[3].items():
        kpss_results[f'Critical Value ({key})'] = value

    print('Results of KPSS Test ----')
    print(kpss_results)


def test_autocorrelation(series, h0_type='c'):

    fig, ax = plt.subplots(2, figsize=(16, 8))
    plot_acf(series, ax=ax[0], lags=40, alpha=0.05)
    plot_pacf(series, ax=ax[1], lags=40, alpha=0.05)
     
    adf_test(series)
    kpss_test(series, h0_type='c')
    print('Autocorrelation plots ----')
    plt.show()


