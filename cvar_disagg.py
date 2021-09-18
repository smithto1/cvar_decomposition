import os
os.chdir('C:\\git\\2021_jobsearch\\medium')


from numpy.random import RandomState
import pandas as pd

from cvar_dayset import CVARDayset

rd = RandomState(1990)

scale = 100
n_assets = 2
n_days = 250

asset_returns = pd.DataFrame(
    rd.normal(loc=0, scale=scale, size=(n_days, n_assets)),
    index=pd.date_range('2020-01-01', freq='B', periods=n_days),
    columns=['asset0', 'asset1']
)

portfolio_0 = asset_returns.mul([100, 100], axis=1)

cvar0 = CVARDayset(portfolio_0)

cvar0.plot_cvar(quantile=.025)



noise = rd.normal(loc=0, scale=scale*.2, size=asset_returns.shape[0])
asset_returns['hedge1'] = asset_returns['asset1'] + noise

portfolio_1= asset_returns.mul([100, 100, -100], axis=1)
cvar1 = CVARDayset(portfolio_1)

cvar0.plot_change(other=cvar1, quantile=.025)


cvar0.plot_change(other=cvar1, quantile=.025, assets=['asset1', 'hedge1'])


    
## save plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
q = .025

with PdfPages('multipage_pdf.pdf') as pdf:
    plt.figure(figsize=(8,6.4))
    cvar0.plot_cvar(q)
    pdf.savefig()
    plt.close()
    
    cvar0.plot_same_days(cvar1, q)
    ylim = plt.ylim()
    pdf.savefig()
    plt.close()
    cvar0.plot_new_days(cvar1, q)
    plt.ylim(ylim)
    pdf.savefig()
    plt.close()
    
    cvar0.plot_same_days(cvar1, q, ['asset1', 'hedge1'])
    ylim = plt.ylim()
    pdf.savefig()
    plt.close()
    cvar0.plot_new_days(cvar1, q, ['asset1', 'hedge1'])
    plt.ylim(ylim)
    pdf.savefig()
    plt.close()
