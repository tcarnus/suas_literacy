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
# + [Frequentist Regression](#Frequentist-Regression)  
# + [Bayesian Regression](#Bayesian-Regression)  
#     + [Unpooled Models](#Unpooled-Models)  
#     + [Hierarchical Models](#Hierarchical-Models)  

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
import time

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

# <markdowncell>

# ---

# <headingcell level=1>

# Frequentist Regression

# <markdowncell>

# ### Entire set

# <codecell>

## scores pre vs post for all

x = df['staard_score_pre']
y = df['staard_score_post']

cm_cmap = cm.get_cmap('hsv')
fig, axes1d = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,8))
fig.subplots_adjust(wspace=0.2)
fig.suptitle('Correlation of pre-test and post-test scores')

sp = axes1d
clr = cm_cmap(0.6)
sp.scatter(x=x,y=y,color=clr,alpha=0.6,edgecolor='#999999')

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
                ,xy=(1,0),xycoords='axes fraction',xytext=(-12,6),textcoords='offset points'
                ,color='green',weight='bold',size=12,ha='right',va='bottom')
sp.set_ylabel('Post-test Score')
sp.set_xlabel('Pre-test Score')
sp.axes.grid(True,linestyle='-',color='lightgrey')

plt.subplots_adjust(top=0.95)
plt.show()  

# <codecell>

## scores pre vs post for test type

## TODO

# <markdowncell>

# ---

# <headingcell level=1>

# Bayesian Regression

# <headingcell level=2>

# Unpooled Models

# <markdowncell>

# Simple linear regression
# 
# ### Entire set

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
    traces = pm.sample(10000, step=pm.NUTS(), start=pm.find_MAP(), progressbar=True)

    pm.traceplot(traces)

# <markdowncell>

# ### Plot regression

# <codecell>

## quick plot of regression (all)

def plot_reg(sp, alpha, beta, sigma, xlims, maxlike=False):  
    x = np.arange(xlims[0], xlims[1])
    y_est = eval('{} + {}*x'.format(alpha, beta))
    if maxlike:    
        sp.plot(x, y_est, linewidth=3, linestyle='dashed', color='#00F5FF', alpha=0.8)
        sp.annotate('{}\nalpha: {:.2f}\nbeta:  {:.2f}\nsigma: {:.2f}'.format(
                    'individual',alpha, beta, sigma)
                ,xy=(1,0),xycoords='axes fraction',xytext=(-12,6),textcoords='offset points'
                ,color='#05B9B0',weight='bold',size=12,ha='right',va='bottom') # #FF5800
    else:
        sp.plot(x, y_est, color='#006266', alpha=0.04)

cm_cmap = cm.get_cmap('hsv')
fig, axes1d = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,8))
fig.subplots_adjust(wspace=0.2)
fig.suptitle('Correlation of pre-test and post-test scores - Bayesian')

sp = axes1d
clr = cm_cmap(0.6)
sp.scatter(x=x,y=y,color=clr,alpha=0.6,edgecolor='#999999')

for i in xrange(1000,5000,10):
    point = traces.point(i)
    plot_reg(sp, point['alpha'], point['beta'], point['sigma'], xlims=[x.min(),x.max()])

plot_reg(sp, traces['alpha'].mean(), traces['beta'].mean(), traces['sigma'].mean()
     ,xlims=[x.min(),x.max()],maxlike=True)  

#plt.show()

# <markdowncell>

# ### Testtype

# <codecell>

## unpooled model for test type

unqvals = np.unique(df.index.get_level_values('test').tolist())
traces_unpooled = {}
max_iter_unpooled = 10000

for testtype in unqvals_tests:

    x = df.query('test=="{}"'.format(testtype))['staard_score_pre']
    y = df.query('test=="{}"'.format(testtype))['staard_score_post']
    
    with pm.Model() as individual_model:

        # priors for intercept, slope and precision - all uninformative
        alpha = pm.Normal('alpha', mu=0, sd=100**2)
        beta = pm.Normal('beta', mu=0, sd=100**2)
        sigma = pm.Uniform('sigma', lower=0, upper=100)

        # Linear model
        y_est = alpha + beta * x

        # Data likelihood
        likelihood = pm.Normal('likelihood', mu=y_est, sd=sigma, observed=y)

        # keep trace
        traces_unpooled[testtype] = pm.sample(max_iter_unpooled, step=pm.NUTS()
                                              ,start=pm.find_MAP() ,progressbar=True)

# <codecell>

burn_in_unpooled = 2000

for testtype in unqvals:
    print('Estimates for: {}'.format(testtype))
    pm.traceplot(traces_unpooled[testtype],figsize=(18,2*3))

# <headingcell level=2>

# Hierarchical Models

# <markdowncell>

# Hierarchical linear regression for test type
# 
# ### Testtype

# <codecell>

## run for test types

# unqvals = np.unique(df.index.get_level_values('test').tolist())
idx_size = unqvals_tests.shape[0]
unqvals_translator = {v:k for k,v in enumerate(unqvals_tests)}
idxs = [unqvals_translator[v] for v in df.index.get_level_values('test').tolist()]

x = df['staard_score_pre']
y = df['staard_score_post']

max_iter = 200000

with pm.Model() as hierarchical_model:

    # hyperpriors for group nodes, all uninformative
    alpha_mu = pm.Normal('alpha_mu', mu=0., sd=100**2)
    alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=100)
    beta_mu = pm.Normal('beta_mu', mu=0., sd=100**2)
    beta_sigma = pm.Uniform('beta_sigma', lower=0, upper=100)
    
    # Intercept for each testtype, distributed around group mean mu_a
    # Above we just set mu and sd to a fixed value while here we
    # plug in a common group distribution for all a and b (which are
    # vectors of length n_testtype).
       
    # priors for alpha, beta and model error, uninformative  
    alpha = pm.Normal('alpha', mu=alpha_mu, sd=alpha_sigma, shape=idx_size)
    beta = pm.Normal('beta', mu=beta_mu, sd=beta_sigma, shape=idx_size)
    sigma = pm.Uniform('sigma', lower=0, upper=100)

    # hierarchical linear model
    y_est = alpha[idxs] + beta[idxs] * x

    # Data likelihood
    likelihood = pm.Normal('likelihood', mu=y_est, sd=sigma, observed=y)

    # keep trace
    traces_hierarchical = pm.sample(max_iter, step=pm.Metropolis()
                                   ,start=pm.find_MAP(), progressbar=True)

# <codecell>

## quick plot of parameters
burn_in = 50000
with pm.Model() as hierarchical_model:
    pm.traceplot(traces_hierarchical,figsize=(18,2*7))

# <markdowncell>

# ### Plot comparison of hierarchical vs unpooled

# <codecell>

def plot_reg_hier(sp, alpha, beta, sigma, xlims, maxlike=False):  
    x = np.arange(xlims[0], xlims[1])
    y_est = eval('{} + {}*x'.format(alpha, beta))
    if maxlike:    
        sp.plot(x, y_est, linewidth=3, linestyle='dashed', color='#FF7538', alpha=0.8)
        sp.annotate('{}\nalpha: {:.2f}\nbeta:  {:.2f}\nsigma: {:.2f}'.format(
                    'hierarchical',alpha, beta, sigma)
                ,xy=(0,1),xycoords='axes fraction',xytext=(12,-6),textcoords='offset points'
                ,color='#FF7538',weight='bold',size=12,ha='left',va='top')
    else:
        sp.plot(x, y_est, color='#661f00', alpha=0.03)


fig, axes1d = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18,8))
fig.subplots_adjust(wspace=0.2)
fig.suptitle('Correlation of pre-test and post-test scores - Bayesian Hierarchical Regression')
cm_cmap = cm.get_cmap('hsv')

for j, (sp, testtype) in enumerate(zip(axes1d,unqvals)):

    x = df.query('test=="{}"'.format(testtype))['staard_score_pre']
    y = df.query('test=="{}"'.format(testtype))['staard_score_post']
    clr = cm_cmap(j/len(unqvals))

    # points
    sp.scatter(x=x,y=y,color=clr,alpha=0.6,edgecolor='#999999')
    sp.annotate(testtype,xy=(0.5,1),xycoords='axes fraction',size=12,ha='center'
                ,xytext=(0,6),textcoords='offset points')

    # unpooled
    for i in np.vectorize(lambda x: int(round(x,0)))\
                    (np.linspace(burn_in_unpooled,max_iter_unpooled,500,endpoint=False)):
        traces = traces_unpooled[testtype]
        point = traces.point(i)
        plot_reg(sp, point['alpha'], point['beta'], point['sigma'], xlims=[x.min(),x.max()])

    plot_reg(sp, traces['alpha'].mean(),traces['beta'].mean(),traces['sigma'].mean()
                    ,xlims=[x.min(),x.max()],maxlike=True)  

    
    # hierarchical
    for i in np.vectorize(lambda x: int(round(x,0)))\
                    (np.linspace(burn_in,max_iter,500,endpoint=False)):
        alpha = traces_hierarchical['alpha'][i][j]
        beta = traces_hierarchical['beta'][i][j]
        sigma = traces_hierarchical['sigma'][i]    # TODO: redundant in this call, refactor
        plot_reg_hier(sp, alpha, beta, sigma, xlims=[x.min(),x.max()])

    plot_reg_hier(sp, traces_hierarchical['alpha'][:,j].mean()
                    ,traces_hierarchical['beta'][:,j].mean()
                    ,traces_hierarchical['sigma'].mean()
                    ,xlims=[x.min(),x.max()],maxlike=True)
    

# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <markdowncell>

# ---
# 
# _copyleft_ **Open Data Science Ireland Meetup Group 2014**  
# <a href='http://www.meetup.com/opendatascienceireland/'>ODSI@meetup.com</a>

