# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:25:45 2023

@author: adayr
"""

import mogptk
import numpy as np

from sklearn.datasets import fetch_openml

def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data['year']
    m = ml_data.data['month']
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs

# load dataset
x, y = load_mauna_loa_atmospheric_co2()

# stop omde to separate train from test
stop = 200

data = mogptk.Data(x, y, name='Mauna Loa')
data.remove_range(start=x[stop])
data.transform(mogptk.TransformDetrend(3))
data.plot()

# create model
model = mogptk.SM(data, Q=3)
model.plot_spectrum(title='SD with random parameters')

method = 'BNSE'
model.init_parameters(method)
model.plot_spectrum(title='PSD with {} initialization'.format(method))