# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# <div style="float:right"><img src='assets/img/Suas_Logo_Header.png'></img></div>
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
#     + [Local Functions](#Local-Functions)
#     + [Import Data](#Import-Data)  
# + [Frequentist Regression](#Frequentist-Regression)  
# + [Bayesian Regression](#Bayesian-Regression)  
#     + [Entire Set](#Entire-Set)  
#     + [Test Type](#Test-Type)  
#     + [Gender](#Gender)
#     + [Age](#Age)
#     + [Date](#Date)

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

# <headingcell level=2>

# Local Functions

# <codecell>

def get_traces_individual(x, y, max_iter=10000):
    """ sample individual model """
    
    with pm.Model() as individual_model:

        # priors for alpha, beta and model error, uninformative  
        alpha = pm.Normal('alpha', mu=0, sd=100**2)
        beta = pm.Normal('beta', mu=0, sd=100**2)
        epsilon = pm.Uniform('epsilon', lower=0, upper=100)

        # individual linear model
        y_est = alpha + beta * x
        likelihood = pm.Normal('likelihood', mu=y_est, sd=epsilon, observed=y)
        traces = pm.sample(max_iter, step=pm.NUTS(), start=pm.find_MAP(), progressbar=True)
    
    return traces


def get_traces_hierarchical(x, y, idxs, max_iter=100000):
    """ sample hierarchical model """
    
    idx_size = len(np.unique(idxs))
    with pm.Model() as hierarchical_model:

        # hyperpriors for group nodes, all uninformative
        alpha_mu = pm.Normal('alpha_mu', mu=0., sd=100**2)
        alpha_sigma = pm.Uniform('alpha_sigma', lower=0, upper=100)
        beta_mu = pm.Normal('beta_mu', mu=0., sd=100**2)
        beta_sigma = pm.Uniform('beta_sigma', lower=0, upper=100)

        # Intercept for each testtype, distributed around group mean mu_a
        # Above mu & sd are fixed value while below we plug in a common 
        # group distribution for all a & b (which are vectors of length idx_size).

        # priors for alpha, beta and model error, uninformative  
        alpha = pm.Normal('alpha', mu=alpha_mu, sd=alpha_sigma, shape=idx_size)
        beta = pm.Normal('beta', mu=beta_mu, sd=beta_sigma, shape=idx_size)
        epsilon = pm.Uniform('epsilon', lower=0, upper=100)

        # hierarchical linear model
        y_est = alpha[idxs] + beta[idxs] * x
        likelihood = pm.Normal('likelihood', mu=y_est, sd=epsilon, observed=y)
        traces = pm.sample(max_iter, step=pm.Metropolis()
                                       ,start=pm.find_MAP(), progressbar=True)
    return traces

# <headingcell level=2>

# Import Data

# <codecell>

## Read cleaned dataset back from db
cnx_sql3 = sqlite3.connect('data/SUAS_data_master_v001_tcc_cleaned.db')
cnx_sql3.text_factory = str
df = pd.read_sql('select * from df_piv', cnx_sql3, index_col=['code','schoolid','gender','test','testtype']
                 , parse_dates=['date_pre','date_post'])
cnx_sql3.close()

df = df.sort_index(axis=1)
df.reset_index(inplace=True)
df.set_index('code',inplace=True)
print(df.shape)
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

## TODO: ## scores pre vs post for test type

# <markdowncell>

# ---

# <headingcell level=1>

# Bayesian Regression

# <codecell>

xy={'x':'staard_score_pre', 'y':'staard_score_post'}

# <headingcell level=2>

# Entire Set

# <markdowncell>

# **(Unpooled)**

# <codecell>

## sample
traces_ind_all = {}
traces_ind_all['all'] = get_traces_individual(df[xy['x']], df[xy['y']], max_iter=10000)

# <codecell>

# plot traces
p = pm.traceplot(traces_ind_all['all'],figsize=(18,1.5*3))
plt.show(p)

# <markdowncell>

# ### Plot regression

# <codecell>

## quick plot of regression 

def plot_reg_bayes(df, xy, traces_ind, traces_hier, feat='no_feat', burn_ind=2000, burn_hier=None):
    """ create plot for bayesian derived regression lines, no groups """
    
    keys = traces_ind.keys()         
    fig, axes1d = plt.subplots(nrows=1, ncols=len(keys), sharex=True, sharey=True, figsize=(8*len(keys),8))
    fig.suptitle('Bayesian hierarchical regression of pre-test vs post-test scores')
    cm_cmap = cm.get_cmap('Set3')

    clrs = {}
    clrs['ind'] = ['#00F5FF','#006266','#00585C']
    clrs['hier'] = ['#FF7538','#661f00','#572610']  
    
    if len(keys) == 1:
        axes1d = [axes1d]
        
    for j, (sp, key) in enumerate(zip(axes1d,keys)):
       
        # scatterplot datapoints and subplot count title
        x = df.loc[:,xy['x']]
        y = df.loc[:,xy['y']]

        if feat != 'no_feat':
            x = df.loc[df[feat] == key,xy['x']]
            y = df.loc[df[feat] == key,xy['y']]               

        sp.scatter(x,y,s=40,color=cm_cmap(j/len(keys)),alpha=0.7,edgecolor='#333333')
        sp.annotate('{} ({} samples)'.format(key,len(x))
            ,xy=(0.5,1),xycoords='axes fraction',size=14,ha='center'
            ,xytext=(0,6),textcoords='offset points')

        # plot regression: individual
        alpha = traces_ind[key]['alpha'][burn_ind:]
        beta = traces_ind[key]['beta'][burn_ind:]
        xfit = np.linspace(x.min(), x.max(), 10)
        
        yfit = alpha[:, None] + beta[:, None] * xfit   # <- yfit for all samples at x in xfit ind
        mu = yfit.mean(0)
        yerr_975 = np.percentile(yfit,97.5,axis=0)
        yerr_025 = np.percentile(yfit,2.5,axis=0)
        
        sp.plot(xfit, mu,linewidth=2, color=clrs['ind'][0], alpha=0.8)
        sp.fill_between(xfit, yerr_025, yerr_975, color=clrs['ind'][2],alpha=0.3)
        sp.annotate('{}\nslope:  {:.2f}\nincpt: {:.2f}'.format(
                                'individual',beta.mean(),alpha.mean())
                ,xy=(1,0),xycoords='axes fraction',xytext=(-12,6),textcoords='offset points'
                ,color=clrs['ind'][1],weight='bold',size=12,ha='right',va='bottom')

        # plot regression: hierarchical
        if traces_hier is not None:    
            alpha = traces_hier['alpha'][burn_hier:,j].T
            beta = traces_hier['beta'][burn_hier:,j].T
            xfit = np.linspace(x.min(), x.max(), 10)
            yfit = alpha[:, None] + beta[:, None] * xfit
            mu = yfit.mean(0)
            yerr_975 = np.percentile(yfit,97.5,axis=0)
            yerr_025 = np.percentile(yfit,2.5,axis=0)

            sp.plot(xfit, mu,linewidth=2, color=clrs['hier'][0], alpha=0.8)
            sp.fill_between(xfit, yerr_025, yerr_975, color=clrs['hier'][2], alpha=0.3)
            sp.annotate('{}\nslope:  {:.2f}\nincpt:  {:.2f}'.format(
                                        'hierarchical',beta.mean(),alpha.mean())
                ,xy=(0,1),xycoords='axes fraction',xytext=(12,-6),textcoords='offset points'
                ,color=clrs['hier'][1],weight='bold',size=12,ha='left',va='top')
        
    plt.show()

# <codecell>

plot_reg_bayes(df,xy,traces_ind_all,None,burn_ind=200)

# <headingcell level=2>

# Test Type

# <markdowncell>

# ### Unpooled

# <codecell>

## run sampling
unqvals_testtype = np.unique(df.testtype)
traces_ind_testtype = {}

for testtype in unqvals_testtype:
    x = df.loc[df.testtype == testtype, xy['x']]
    y = df.loc[df.testtype == testtype, xy['y']]
    traces_ind_testtype[testtype] = get_traces_individual(x, y, max_iter=10000)

# <codecell>

## view parameters
# for testtype in unqvals_testtype:
#     print('Estimates for: {}'.format(testtype))
#     pm.traceplot(traces_ind_testtype[testtype],figsize=(18,1.5*3))

# <markdowncell>

# ### Hierarchical

# <codecell>

## run sampling
unqvals_translator = {v:k for k,v in enumerate(unqvals_testtype)}
idxs = [unqvals_translator[v] for v in df.testtype]
traces_hier_testtype = get_traces_hierarchical(df[xy['x']], df[xy['y']], idxs, max_iter=100000)

# <codecell>

## view parameters
# with pm.Model() as hierarchical_model:
#     pm.traceplot(traces_hier_testtype,figsize=(16,1.5*7))

# <markdowncell>

# ### Plot comparison of hierarchical vs unpooled

# <codecell>

plot_reg_bayes(df, xy ,traces_ind_testtype, traces_hier_testtype
               , feat='testtype', burn_ind=2000, burn_hier=50000)

# <headingcell level=2>

# Gender

# <markdowncell>

# ### Unpooled

# <codecell>

## unpooled model for gender
unqvals_gender = np.unique(df.gender)
traces_unpooled = {}
for gender in unqvals_gender:
    x = df.loc[df.gender == gender, 'staard_score_pre']
    y = df.loc[df.gender == gender, 'staard_score_post']
    traces_unpooled[gender] = get_traces_unpooled(x, y, max_iter=10000)

# <codecell>

### view parameters
for gender in unqvals_gender:
    print('Estimates for: {}'.format(gender))
    pm.traceplot(traces_unpooled[gender],figsize=(18,2*3))

# <markdowncell>

# ### Hierarchical

# <codecell>

## run traces
unqvals_translator = {v:k for k,v in enumerate(unqvals_gender)}
idxs = [unqvals_translator[v] for v in df.gender]
x = df['staard_score_pre']
y = df['staard_score_post']
traces_hierarchical = get_traces_hierarchical(x, y, idxs, max_iter=10000)

# <codecell>

## quick plot of parameters
with pm.Model() as hierarchical_model:
    pm.traceplot(traces_hierarchical,figsize=(18,2*7))

# <markdowncell>

# ### Plot comparison of hierarchical vs unpooled

# <codecell>

fig, axes1d = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18,8))
fig.subplots_adjust(wspace=0.2)
fig.suptitle('Correlation of pre-test vs post-test scores - Bayesian Hierarchical Regression')
cm_cmap = cm.get_cmap('Set3')

for j, (sp, gender) in enumerate(zip(axes1d,unqvals_gender)):

    x = df.loc[df.gender == gender, 'staard_score_pre']
    y = df.loc[df.gender == gender, 'staard_score_post']
    
    clr = cm_cmap(j/len(unqvals))

    # datapoints and subplot titles
    sp.scatter(x=x,y=y,s=40,color=clr,alpha=0.7,edgecolor='#333333')
    sp.annotate('{} ({} samples)'.format(gender,len(x))
                ,xy=(0.5,1),xycoords='axes fraction',size=14,ha='center'
                ,xytext=(0,6),textcoords='offset points')
    
    # regression fit independent
    plot_reg(sp,traces_unpooled[gender],x)
    
    # regression fit hierarchical
    plot_reg(sp,traces_hierarchical,x,usage='hierarchical',col=j)
        

# <headingcell level=2>

# Age

# <markdowncell>

# Bin age into 1-yearly sub groups

# <codecell>

# Bin age
df['binned_age_pre'] = df['age_pre'].apply(lambda x: round(x,0))

# <codecell>

## unpooled model for binned age




unqvals = np.unique(df.binned_age_pre)
traces_unpooled = {}
max_iter_unpooled = 10000

for binned_age in unqvals:

    x = df.loc[df.binned_age_pre == binned_age, 'staard_score_pre']
    y = df.loc[df.binned_age_pre == binned_age, 'staard_score_post']
    
    with pm.Model() as individual_model:

        # priors for intercept, slope and precision - all uninformative
        alpha = pm.Normal('alpha', mu=0, sd=100**2)
        beta = pm.Normal('beta', mu=0, sd=100**2)
        epsilon = pm.Uniform('epsilon', lower=0, upper=100)

        # Linear model
        y_est = alpha + beta * x
        likelihood = pm.Normal('likelihood', mu=y_est, sd=epsilon, observed=y)
        traces_unpooled[binned_age] = pm.sample(max_iter_unpooled, step=pm.NUTS()
                                              ,start=pm.find_MAP() ,progressbar=True)

# <codecell>

### view parameters
for binned_age in unqvals:
    print('Estimates for: {}'.format(binned_age))
    pm.traceplot(traces_unpooled[binned_age],figsize=(18,2*3))

# <markdowncell>

# ### Hierarchical

# <codecell>

## run traces
unqvals = np.unique(df.binned_age_pre)
unqvals_translator = {v:k for k,v in enumerate(unqvals)}
idxs = [unqvals_translator[v] for v in df.binned_age_pre]
x = df['staard_score_pre']
y = df['staard_score_post']
traces_hierarchical = get_traces_hierarchical(x, y, idxs, max_iter=200000)

# <codecell>

## quick plot of parameters
with pm.Model() as hierarchical_model:
    pm.traceplot(traces_hierarchical,figsize=(18,2*7))

# <markdowncell>

# ### Plot comparison of hierarchical vs unpooled

# <codecell>

unqvals_age = np.unique(df.binned_age_pre)
fig, axes1d = plt.subplots(nrows=1, ncols=len(unqvals), sharex=True, sharey=True, figsize=(18,3.5))
fig.subplots_adjust(wspace=0.2)
fig.suptitle('Correlation of pre-test vs post-test scores - Bayesian Hierarchical Regression')
cm_cmap = cm.get_cmap('Set3')

for j, (sp, binned_age) in enumerate(zip(axes1d,unqvals)):

    x = df.loc[df.binned_age_pre == binned_age, 'staard_score_pre']
    y = df.loc[df.binned_age_pre == binned_age, 'staard_score_post']
    clr = cm_cmap(j/len(unqvals))

    # datapoints and subplot titles
    sp.scatter(x=x,y=y,s=40,color=clr,alpha=0.7,edgecolor='#333333')
    sp.annotate('{} ({} samples)'.format(binned_age,len(x))
                ,xy=(0.5,1),xycoords='axes fraction',size=14,ha='center'
                ,xytext=(0,6),textcoords='offset points')

    plot_reg(sp,traces_unpooled[binned_age],x,burn_ratio=0.25, ipoints=10)
    plot_reg(sp,traces_hierarchical,x,usage='hierarchical',col=j,burn_ratio=0.5, ipoints=10)
        

# <codecell>


# <codecell>


# <markdowncell>

# ---
# 
# _copyleft_ **Open Data Science Ireland Meetup Group 2014**  
# <a href='http://www.meetup.com/opendatascienceireland/'>ODSI@meetup.com</a>

