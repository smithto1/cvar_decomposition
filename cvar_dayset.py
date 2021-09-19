
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.pyplot import Subplot
import numpy as np
import pandas as pd


class CVARDayset:
    def __init__(self, pnl_vectors:pd.DataFrame, interpolation:str='lower'):
        self.pnl_vectors = pnl_vectors
        self.interpolation = interpolation
        self.color = 'skyblue'
        self.dt_format = '%d-%b-%y'

    @property
    def total_pnl(self):
        return self.pnl_vectors.sum(axis=1)

    def var(self, q:float):
        return self.total_pnl.quantile(q, interpolation=self.interpolation)

    def cvar_index(self, q:float):
        return self.pnl_vectors[self.total_pnl < self.var(q)].index
    
    def cvar(self, q:float, days:bool=True, assets:bool=True):
        res = self.pnl_vectors.reindex(self.cvar_index(q))
        if assets:
            res = res.sum(axis=1)
        if days:
            res = res.mean(axis=0)
        return res

    def _index_sets(self, other, q:float):
        sidx = self.cvar_index(q)
        oidx = other.cvar_index(q)
        return sidx.difference(oidx), sidx.intersection(oidx)

    @staticmethod
    def _srs(vals, idxs):
        return pd.concat([pd.Series(v, i) for v,i in zip(vals, idxs)])

    def _assets(self, other, assets):
        if assets is None:
            assets = self.pnl_vectors.columns.union(other.pnl_vectors.columns)
            return assets, '', ''
        else:
            return assets, '\nAssets: {}'.format(', '.join(assets)), ' Contribution'

    @staticmethod
    def _widest_lims(subplots:List[Subplot], x_y:str):
        lims = []
        for sub in subplots:
            plt.axes(sub)
            lims.append(getattr(plt, x_y)())
        final = (
            min([tup[0] for tup in lims]),
            max([tup[1] for tup in lims])
        )
        for sub in subplots:
            plt.axes(sub)
            getattr(plt, x_y)(final)
        return None
    
    @staticmethod
    def _new_fig():
        fig = plt.figure(figsize=(6.5,6.5))
        gs = fig.add_gridspec(5,1)
        sub0 = fig.add_subplot(gs[:1,:])
        sub1 = fig.add_subplot(gs[1:,:])
        return sub0, sub1
    
    def _plot_date_text(self, title, x, index, index_sets):
        plt.title(title)
        plt.bar(x=x, height=[0])
        plt.ylim((0,1))
        
        plt.yticks([0, 1], ['Remain', 'Change'])
        plt.xticks([], [])
        y = self._srs([.99, .01], index_sets).reindex(index)
        for xi, yi, si in zip(x, y, index):
            va = 'bottom' if yi == y.min() else 'top'
            plt.text(x=xi, y=yi, s=si.strftime(self.dt_format), 
                      horizontalalignment='center', 
                      verticalalignment=va,rotation=90)
    
    def plot_same_days(self, other, q:float, assets:List[str]=None,
                       subplots:Tuple[Subplot, Subplot]=None):
        
        asssets, assets_title, contribution_str = self._assets(other, assets)
        
        heights = self.cvar(q, days=False, assets=False)
        heights = heights.reindex(columns=assets).sum(axis=1).sort_values()
        
        x = np.arange(heights.shape[0])
        index_sets = self._index_sets(other, q)
        
        # create axes if None given
        sub0, sub1 = self._new_fig() if subplots is None else subplots

        # top plot with text
        plt.axes(sub0)
        title = 'Apply New Positions to Old CVaR Days' + assets_title
        self._plot_date_text(title, x, heights.index, index_sets)
        
        # bottom plot with bars
        oth_bars = other.pnl_vectors.reindex(columns=assets).sum(axis=1).reindex(heights.index)
        sub1.bar(x=x, height=oth_bars, width=1, color=self.color, 
                 edgecolor='black')
        sub1.bar(x=x, height=heights, width=1, fill=False, edgecolor='black')
        plt.axes(sub1)
        plt.xticks([], [])
        
        # summary box
        cvar0 = f'{int(heights.mean()):,}'
        cvar1 = f'{int(oth_bars.mean()):,}'
        s = f'Old CVaR({q*100}%){contribution_str}: {cvar0}' 
        s = s + f'\nNew Positions on Old CVaR({q*100}%) Days: {cvar1}'
        x,y = max(x), min(plt.ylim())*.95
        plt.text(x+.5,y,s, verticalalignment='bottom', horizontalalignment='right')
        plt.tight_layout()
        
        return None
    
    def plot_new_days(self, other, q:float, assets:List[str]=None,
                      subplots:Tuple[Subplot, Subplot]=None):
        
        asssets, assets_title, contribution_str = self._assets(other, assets)
        
        heights = other.cvar(q, days=False, assets=False)
        heights = heights.reindex(columns=assets).sum(axis=1).sort_values()
        
        x = np.arange(heights.shape[0])
        index_sets = other._index_sets(self, q)
        
        # create axes if None given
        sub0, sub1 = self._new_fig() if subplots is None else subplots

        plt.axes(sub0)
        title = 'New Positions on New CVaR Days' + assets_title
        self._plot_date_text(title, x, heights.index, index_sets)

        # bottom plot with bars
        plt.axes(sub1)
        plt.bar(x=x, height=heights, width=1, color=self.color, edgecolor='black')
        plt.xticks([], [])
        cvar = f'{int(heights.mean()):,}'
        s = f'New CVaR({q*100}%){contribution_str}: {cvar}'
        x,y = max(x), min(plt.ylim())*.95
        plt.text(x+.5,y,s, verticalalignment='bottom', horizontalalignment='right')
        plt.tight_layout()
        
        return None
    
    def plot_change(self, other, q:float, assets:List[str]=None):
        fig = plt.figure(figsize=(16,8))
        gs = fig.add_gridspec(5,2)
        sub0 = fig.add_subplot(gs[:1,0])
        sub1 = fig.add_subplot(gs[1:,0])
        self.plot_same_days(other, q, assets, (sub0, sub1))

        sub2 = fig.add_subplot(gs[:1,1])
        sub3 = fig.add_subplot(gs[1:,1])
        self.plot_new_days(other, q, assets, (sub2, sub3))
        
        self._widest_lims([sub1, sub3], 'ylim')
        return None

    def plot_cvar(self, q:float, assets:List[str]=None):
        
        asssets, assets_title, contribution_str = self._assets(self, assets)
        
        heights = self.cvar(q, days=False, assets=False)
        heights = heights.reindex(columns=assets).sum(axis=1).sort_values()
        
        x = np.arange(heights.shape[0])
        
        plt.title('CVaR Days' + assets_title)
        plt.bar(x=x, height=heights, width=1, fill=None, edgecolor='black')
        plt.xticks([], [])
        y = .05*min(plt.ylim())
        for xi, si in zip(x, heights.index):
            plt.text(x=xi, y=y, s=si.strftime(self.dt_format), 
                     horizontalalignment='center', 
                     verticalalignment='top', rotation=90)
        
        cvar = f'{int(heights.mean()):,}'
        s = f'CVaR({q*100}%){contribution_str}: {cvar}'
        x,y = max(x), min(plt.ylim())*.95
        plt.text(x+.5,y,s, verticalalignment='bottom', horizontalalignment='right')
        plt.tight_layout()
        return None
