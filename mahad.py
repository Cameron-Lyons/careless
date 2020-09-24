'''Find and graph Mahalanobis Distance (D) and flag potential outliers

Takes a matrix of item responses and computes Mahalanobis D. Can additionally return a
vector of binary outlier flags.
Mahalanobis distance is calculated using a function which supports missing values.'''

import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def mahad(x, plot=True, flag=False, confidence=0.99, drop_na=True):
    if not drop_na:
        if any(x.is_null()):
            raise Exception("Some values are NA. Mahalanobis distance was not computed. \
                                         Use drop_na=TRUE to use available cases.")

    #remove rows with all NA and issue warning
    complete_na = x.is_null()
    if any(complete_na):
        raise Exception("Some cases contain only NA values. The Mahalanobis distance will be calculated using available cases.")
    x_filtered = x.dropna()

    maha_data = pd.as_numeric(mahalanobis(x))
    maha_data_merge = np.resize(complete_na, len(x_filtered))
    maha_data_merge[complete_na == False] = maha_data

    if flag:
        cut = chi2.ppf(confidence, len(maha_data))
        flagged = (maha_data_merge > cut)
        return maha_data_merge, flagged

    else:
        return maha_data_merge
