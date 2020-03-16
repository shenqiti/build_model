'''
By: shenqiti
2019/3/16
'''

import warnings
import collections

import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from sklearn.utils import check_array


def dm_test(e1, e2, alternative='two_sided', h=1, power=2):
    """
    The Diebold-Mariano test for predictive accuracy.
    The DM test compares the forecast accuracy of different forecasting methods. This implementation is designed
    according to R implementation which could be found here: https://pkg.robjhyndman.com/forecast/reference/dm.test.html
    :param e1: forecast errors from the first method
    :param e2: forecast errors from the second method
    :param alternative: str specifying the alternative hypothesis, 'two_sided' (default one), 'less' or 'greater'
    :param h: forcasting horizon used in calculating errors (e1, e2)
    :param power: power used in the loss function (usually 1 or 2)
    :return: named tuple containing DM statistic and p-value
    Examples
    --------
    dm_test_result(dm_stat=1.023019805162665, p_value=0.30879181140619405)
    """
    alternatives = ['two_sided', 'less', 'greater']
    if alternative not in alternatives:
        raise ValueError(f"alternative must be one of {alternatives}")

    e1 = check_array(e1, ensure_2d=False)
    e2 = check_array(e2, ensure_2d=False)

    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = d.shape[0]
    d_cov = sm.tsa.acf(d, fft=True, nlags=h - 1)
    d_var = (d_cov[0] + 2 * d_cov[1:].sum()) / n

    if d_var > 0:
        dm_stat = np.mean(d) / np.sqrt(d_var)
    elif h == 1:
        raise ValueError('Variance of DM statistic is zero')
    else:
        warnings.warn('Variance is negative, using horizon h=1', RuntimeWarning)
        return dm_test(e1, e2, alternative=alternative, h=1, power=power)

    # The corrected statistic suggested by HLN
    k = ((n + 1 - 2 * h + h / n * (h - 1)) / n) ** 0.5
    dm_stat *= k

    if alternative == 'two_sided':
        p_value = 2 * t.cdf(-abs(dm_stat), df=n - 1)
    else:
        p_value = t.cdf(dm_stat, df=n - 1)
        if alternative == 'greater':
            p_value = 1 - p_value

    dm_test_result = collections.namedtuple('dm_test_result', ['dm_stat', 'p_value'])
    return dm_test_result(dm_stat=dm_stat, p_value=p_value)
