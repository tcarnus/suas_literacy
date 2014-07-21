# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# <div style="float:right"><img src='files/assets/img/Suas_Logo_Header.png'></img></div>
# 
# ## Suas Educational Development - Literacy Intervention Investigation
# 
# # 02 Initial Exploration
# 
# Contact: Adelaide Nic Chartaigh [adelaide@suas.ie]("mailto:adelaide@suas.ie")  
# Author: Jonathan Sedar [jon.sedar@applied.ai]("mailto:jon.sedar@applied.ai")  
# Date: Spring / Summer 2014
# 
# #Contents
# 
# + [Setup](#Setup)  
# + [Import Data](#Import-Data) 
# + [Visualisation and Single Feature Regression](#Visualisation-and-Single-Feature-Regression)

# <headingcell level=1>

# Setup

# <codecell>

## Interactive magics
%matplotlib inline
%qtconsole --colors=linux --ConsoleWidget.font_size=10 --ConsoleWidget.font_family='Consolas'

# <codecell>

## Libraries and global options
from __future__ import division, print_function
import os
import sys
import re
import string

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import sqlite3
from sklearn import linear_model
import pymc as pm

#from sklearn.covariance import MinCovDet
#from sklearn.preprocessing import Normalizer
#from collections import OrderedDict

# Set some default pandas and plotting formatting
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.notebook_repr_html', True)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.precision', 5)
pd.set_option('display.max_colwidth', 50)

plt.rcParams['figure.figsize'] = 6, 6

# handle temporary deprecation warning in pandas for describe()
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)

remove_punct_map = dict.fromkeys(map(ord, string.punctuation))

# <headingcell level=1>

# Import Data

# <codecell>

## Read cleaned dataset back from db
cnx_sql3 = sqlite3.connect('data/SUAS_data_master_v001_tcc_cleaned.db')
cnx_sql3.text_factory = str
df = pd.read_sql('select * from df_piv', cnx_sql3, index_col=['code','schoolid','gender','test','testtype']
                 , parse_dates=['date_pre','date_post'])
cnx_sql3.close()

print(df.shape)
df.head()

# <markdowncell>

# ### Calc some deltas for convenience

# <codecell>

df['date_delta'] = df['date_post'] - df['date_pre']
df['age_delta'] = df['age_post'] - df['age_pre']
df['raw_score_delta'] = df['raw_score_post'] - df['raw_score_pre']
df['staard_score_delta'] = df['staard_score_post'] - df['staard_score_pre']

df = df.sort_index(axis=1)
df.head()

# <headingcell level=1>

# Visualisation and Single Feature Regression

# <codecell>

## scores pre vs post for all

x = df['staard_score_pre']
y = df['staard_score_post']

cm_cmap = cm.get_cmap('hsv')
fig, axes1d = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12,6))
fig.subplots_adjust(wspace=0.2)
fig.suptitle('Correlation of pre-test and post-test scores')

sp = axes1d
clr = cm_cmap(0)
sp.scatter(x=x,y=y,color=clr,alpha=0.5,edgecolor='#999999')

# fit and plot a new linear model
regr = linear_model.LinearRegression()
regr.fit(pd.DataFrame(x),pd.DataFrame(y))
x_prime = np.linspace(x.min(),x.max(),num=10)[:,np.newaxis]
y_hat = regr.predict(x_prime)
sp.plot(x_prime,y_hat,linewidth=2,linestyle='dashed',color='green',alpha=0.8)

ss_res = regr.residues_[0]
ss_tot = np.sum((y - np.mean(y))**2)
rsq = 1 - (ss_res/ss_tot)

sp.annotate('R^2:  {:.2f}\nCoef: {:.2f}\nIntr: {:.2f}'.format(rsq,regr.coef_[0][0],regr.intercept_[0])
                ,xy=(1,0),xycoords='axes fraction'
                ,xytext=(-12,6),textcoords='offset points',color='green',weight='bold'
                ,size=12,ha='right',va='bottom')
sp.set_ylabel('Post-test Score')
sp.set_xlabel('Pre-test Score')
sp.axes.grid(True,linestyle='-',color='lightgrey')

plt.subplots_adjust(top=0.85)
plt.show()  

# <codecell>

## scores pre vs post for test type

## TODO

# <markdowncell>

# ---

# <headingcell level=1>

# Bayesian Regression

# <markdowncell>

# ## Unpooled approach 
# ... for comparison with Frequentist

# <codecell>

## try unpooled model first for all

x = df['staard_score_pre']
y = df['staard_score_post']

with pm.Model() as individual_model:

    # priors for intercept, slope and precision - all uninformative
    alpha = pm.Normal('alpha', mu=0, sd=100**2)
    beta = pm.Normal('beta', mu=0, sd=100**2)
    sigma = pm.Uniform('sigma', lower=0, upper=100)

    # Linear model
    y_est = alpha + beta * x

    # Data likelihood
    likelihood = pm.Normal('event like', mu=y_est, sd=sigma, observed=y)

    # keep trace
    traces = pm.sample(5000, step=pm.NUTS(), start=pm.find_MAP(), progressbar=True)

    pm.traceplot(traces)

# <codecell>

## quick plot of regression

def plot_reg(sp, alpha, beta, sigma, xlims, maxlike=False):  
    x = np.arange(xlims[0], xlims[1])
    y_est = eval('{} + {}*x'.format(alpha, beta))
    if maxlike:    
        sp.plot(x, y_est, linewidth=3, linestyle='dashed', color='darkorange', alpha=0.8)
        sp.annotate('alpha: {:.2f}\nbeta:  {:.2f}\nsigma: {:.2f}'.format(alpha, beta, sigma)
                ,xy=(1,0),xycoords='axes fraction',xytext=(-12,6),textcoords='offset points'
                ,color='darkorange',weight='bold',size=12,ha='right',va='bottom')
    else:
        sp.plot(x, y_est, color='#333333', alpha=0.05)

cm_cmap = cm.get_cmap('hsv')
fig, axes1d = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12,6))
fig.subplots_adjust(wspace=0.2)
fig.suptitle('Correlation of pre-test and post-test scores - Bayesian')

sp = axes1d
clr = cm_cmap(0)
sp.scatter(x=x,y=y,color=clr,alpha=0.5,edgecolor='#999999')
plot_reg(sp, traces['alpha'].mean(), traces['beta'].mean(), traces['sigma'].mean()
         ,xlims=[x.min(),x.max()],maxlike=True)  

for i in xrange(1000,5000,50):
    point = traces.point(i)
    plot_reg(sp, point['alpha'], point['beta'], point['sigma'], xlims=[x.min(),x.max()])

#plt.show()

# <codecell>

## try unpooled model first for test type

unqvals_tests = np.unique(df.index.get_level_values('test').tolist())
traces_unpooled = {}

for testtype in unqvals_tests:

    x = df.query('test=="{}"'.format(testtype))['staard_score_pre']
    y = df.query('test=="{}"'.format(testtype))['staard_score_post']
    
    with pm.Model() as individual_model:
        
        # intercept prior (variance == sd**2) and slope prior
        alpha = pm.Normal('alpha', mu=0, sd=100**2)
        beta = pm.Normal('beta', mu=0, sd=100**2)
    
        # Model error prior
        eps = pm.Uniform('eps', lower=0, upper=100)
    
        # Linear model
        y_est = alpha + beta * x
    
        # Data likelihood
        likelihood = pm.Normal('event like', mu=y_est, sd=eps, observed=y)
        
        # keep trace
        traces_unpooled[testtype] = pm.sample(2000, step=pm.NUTS(), start=pm.find_MAP(), progressbar=True)
        
        # this is more robust to outliers you need stats model installed
        # with pm.Model() as model_robust:
        #    family = pm.glm.families.T()
  

# <codecell>

for testtype in unqvals_tests:
    print('Estimates for: {}'.format(testtype))
    tp = pm.traceplot(traces_unpooled[testtype])
    tp.show()

# <codecell>

def plot_breg(formula, x_min, x_max):  
    x = np.arange(x_min, x_max)
    y_est = eval(formula)
    plt.plot(x, y_est)  
    
plt.scatter(x,y)
plot_breg('23 + 0.79*x', x.min(), x.max())
plt.show()

# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <markdowncell>

# ---
# 
# _copyleft_ **Open Data Science Ireland Meetup Group 2014**  
# <a href='http://www.meetup.com/opendatascienceireland/'>ODSI@meetup.com</a>

