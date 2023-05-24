from statsmodels.tsa.stattools import adfuller


def check_stationarity(ts, cfg='full'):
    """
    :param ts: target value from datasets
    :param cfg: type full or smth else
    :return: info about stationary
    """
    dftest = adfuller(ts)
    adf = dftest[0]
    pvalue = dftest[1]
    critical_value = dftest[4]['5%']

    if cfg == 'full':
        print('ADF Statistic: %f' % dftest[0])
        print('p-value: %f' % dftest[1])
        print('Critical Values:')

    for key, value in dftest[4].items():
        if cfg == 'full':
            print('\t%s: %.3f' % (key, value))
    if (pvalue < 0.05) and (adf < critical_value):
        print('The TS is stationary')
    else:
        print('The TS is NOT stationary')


def make_diff_n_degree(target, n):
    diff_ts = target

    for i in range(n):
        diff_ts = diff_ts.diff(periods=1).dropna()

    return diff_ts




