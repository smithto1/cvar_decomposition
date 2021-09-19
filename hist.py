
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = 4
q = .075

x = np.linspace(t.ppf(0.005, df),
                t.ppf(0.995, df), 100)

pdf = t.pdf(x, df)


with PdfPages('histogram.pdf') as pdf_file:
    fig = plt.figure(figsize=(4,3))
    plt.plot(x, pdf)
    plt.xticks([],[])
    plt.yticks([],[])
    ylim = plt.ylim()
    plt.ylim((0, ylim[1]))
    
    idx = np.where(x > t.ppf(q, df))[0].min()
    
    xi = x[:idx]
    yi = pdf[:idx]
    
    plt.fill_between(xi, 0, yi)
    
    plt.hlines(yi.max(), xi.min(), xi.max())
    plt.vlines(xi.min(), yi.max()-.02, yi.max()+.02)
    plt.vlines(xi.max(), yi.max()-.02, yi.max()+.02)
    
    plt.text(xi.mean(), yi.max()+.06, '$CVaR_q=$', horizontalalignment='center')
    plt.text(xi.mean(), yi.max()+.03, '$E[r|r<VaR_q]$', horizontalalignment='center')
    
    plt.xticks([xi.max()], ['$VaR_q$'])
    plt.tight_layout()
    
    pdf_file.savefig()
    plt.close()
