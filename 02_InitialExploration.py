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

def get_traces_unpooled(x, y, max_iter=10000):
    """ sample unpooled """
    
    with pm.Model() as hierarchical_model:

        # priors for alpha, beta and model error, uninformative  
        alpha = pm.Normal('alpha', mu=0, sd=100**2)
        beta = pm.Normal('beta', mu=0, sd=100**2)
        epsilon = pm.Uniform('epsilon', lower=0, upper=100)

        # hierarchical linear model
        y_est = alpha + beta * x
        likelihood = pm.Normal('likelihood', mu=y_est, sd=epsilon, observed=y)
        traces = pm.sample(max_iter, step=pm.NUTS(), start=pm.find_MAP(), progressbar=True)
    
    return traces


def get_traces_hierarchical(x, y, idxs, max_iter=100000):
    """ sample hierarchical """
    
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


def plot_reg(sp, traces, x_actual, burn_ratio=0.25, ipoints=1000, usage='individual', col=0):  
    """ create plot for bayesian derived regression lines """
    
    burn_start = int(burn_ratio*len(traces))
    x = np.arange(x_actual.min(),x_actual.max())
    
    clrs = {}
    clrs['individual'] = ['#00F5FF','#006266','#00585C']
    clrs['hierarchical'] = ['#FF7538','#661f00','#572610']
        
    # draw credible interval
    for i in np.vectorize(lambda x: int(x))\
                    (np.linspace(burn_start,len(traces),ipoints,endpoint=False)):
        if usage == 'individual':
            point = traces.point(i)
            sp.plot(x, eval('{} + {}*x'.format(point['alpha'], point['beta']))
                ,color=clrs[usage][1], alpha=0.02)
        else:
            sp.plot(x, eval('{} + {}*x'.format(traces['alpha'][i][j], traces['beta'][i][j]))
                ,color=clrs[usage][1], alpha=0.02)            

    # draw mean and annotation
    if usage == 'individual':
        sp.plot(x, eval('{} + {}*x'.format(
                    traces['alpha'].mean(), traces['beta'].mean()))
                    ,linewidth=2, color=clrs[usage][0], alpha=0.8)

        sp.annotate('{}\nalpha: {:.2f}\nbeta:  {:.2f}'.format(usage
                ,traces['alpha'].mean(),traces['beta'].mean())
                ,xy=(1,0),xycoords='axes fraction',xytext=(-12,6),textcoords='offset points'
                ,color=clrs[usage][1],weight='bold',size=12,ha='right',va='bottom')

    else:
        sp.plot(x, eval('{} + {}*x'.format(
                    traces['alpha'][:,col].mean(), traces['beta'][:,col].mean()))
                    ,linewidth=2, color=clrs[usage][0], alpha=0.8)
    
        sp.annotate('{}\nalpha: {:.2f}\nbeta:  {:.2f}'.format(usage
                ,traces['alpha'][:,col].mean(),traces['beta'][:,col].mean())
                ,xy=(0,1),xycoords='axes fraction',xytext=(12,-6),textcoords='offset points'
                ,color=clrs[usage][1],weight='bold',size=12,ha='left',va='top')

        
        
        
        

        
        
# def plot_MCMC_model(ax, xdata, ydata, trace):
#     """Plot the linear model and 2sigma contours"""
#     ax.plot(xdata, ydata, 'ok')

#     alpha, beta = trace[:2]
#     xfit = np.linspace(-20, 120, 10)
#     yfit = alpha[:, None] + beta[:, None] * xfit
#     mu = yfit.mean(0)
#     sig = 2 * yfit.std(0)

#     ax.plot(xfit, mu, '-k')
#     ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
        
        
        

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

# <codecell>

## scores pre vs post for test type

## TODO

# <markdowncell>

# ---

# <headingcell level=1>

# Bayesian Regression

# <headingcell level=2>

# Entire Set

# <markdowncell>

# ### Unpooled

# <codecell>

## sample
x = df['staard_score_pre']
y = df['staard_score_post']
traces_unpooled = get_traces_unpooled(x, y, max_iter=1000)

# <codecell>

## plot traces
p = pm.traceplot(traces_unpooled,figsize=(18,2*3))
plt.show(p)

# <markdowncell>

# ### Plot regression

# <codecell>

## quick plot of regression (all)

xy = {'x':'staard_score_pre', 'y':'staard_score_post'}
feat='no_feat'
traces_hier = None

#def plot_reg_single(df, xy={}, traces_unpooled={}, feat='no_feat', traces_hierarchical=None):  
""" create plot for bayesian derived regression lines, no groups """


fig, axes1d = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6*1,6))
fig.suptitle('Bayesian hierarchical regression of pre-test vs post-test scores')
cm_cmap = cm.get_cmap('Set3')

# for j, (sp, key) in enumerate(zip(axes1d,keys)):

sp = axes1d

clr = cm_cmap(0)
x = df.loc[:,xy['x']]
y = df.loc[:,xy['y']]

if feat != 'no_feat':
    x = df.loc[df.loc[feat] == key,xy['x']]
    y = df.loc[df.loc[feat] == key,xy['y']]               
    
# datapoints and subplot titles
sp.scatter(x,y,s=40,color=clr,alpha=0.7,edgecolor='#333333')
sp.annotate('{} ({} samples)'.format('all',len(x))
            ,xy=(0.5,1),xycoords='axes fraction',size=14,ha='center'
            ,xytext=(0,6),textcoords='offset points')

# unpooled
alpha = traces_unpooled['alpha'][burn_in:]
beta = traces_unpooled['beta'][burn_in:]

xfit = np.linspace(x.min(), x.max(), 10)
yfit = alpha[:, None] + beta[:, None] * xfit   # <- gnarly vectorized code!
mu = yfit.mean(0)
sig = 2 * yfit.std(0)

sp.plot(xfit, mu,linewidth=2, color='#00F5FF', alpha=0.8)
sp.fill_between(xfit, mu - sig, mu + sig, color='#006266',alpha=0.1)
    
    
if traces_hier is not None:    
    # heirarchical
    alpha = traces_hier[key]['alpha'][burn_in:].mean()
    beta = traces_heir[key]['beta'][burn_in:].mean()

    xfit = np.linspace(x.min(), x.max(), 10)
    yfit = alpha + beta * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    #     ax.plot(xfit, mu, '-k')
    #     ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    sp.plot(xfit, yfit,linewidth=2, color='#00F5FF', alpha=0.8)
    sp.fill_between(xfit, yfit - sig, yfit + sig, color='#006266',alpha=0.6)
    

    
#        plot_reg(sp,traces_hierarchical,x,usage='hierarchical',col=j,burn_ratio=0.5, ipoints=10)

plt.show()





#keys = traces_unpooled.keys()
# fig, axes1d = plt.subplots(nrows=1, ncols=len(keys), sharex=True, sharey=True, figsize=(4*len(keys),4))




# p = plot_reg2(df, {'x':'staard_score_pre', 'y':'staard_score_pre'}, traces_unpooled, feat=False)
# plt.show(p)



    
#     burn_start = int(burn_ratio*len(traces))
#     x = np.arange(x_actual.min(),x_actual.max())
    
#     clrs = {}
#     clrs['individual'] = ['#00F5FF','#006266','#00585C']
#     clrs['hierarchical'] = ['#FF7538','#661f00','#572610']
        
#     # draw credible interval
#     for i in np.vectorize(lambda x: int(x))\
#                     (np.linspace(burn_start,len(traces),ipoints,endpoint=False)):
#         if usage == 'individual':
#             point = traces.point(i)
#             sp.plot(x, eval('{} + {}*x'.format(point['alpha'], point['beta']))
#                 ,color=clrs[usage][1], alpha=0.02)
#         else:
#             sp.plot(x, eval('{} + {}*x'.format(traces['alpha'][i][j], traces['beta'][i][j]))
#                 ,color=clrs[usage][1], alpha=0.02)            

#     # draw mean and annotation
#     if usage == 'individual':
#         sp.plot(x, eval('{} + {}*x'.format(
#                     traces['alpha'].mean(), traces['beta'].mean()))
#                     ,linewidth=2, color=clrs[usage][0], alpha=0.8)

#         sp.annotate('{}\nalpha: {:.2f}\nbeta:  {:.2f}'.format(usage
#                 ,traces['alpha'].mean(),traces['beta'].mean())
#                 ,xy=(1,0),xycoords='axes fraction',xytext=(-12,6),textcoords='offset points'
#                 ,color=clrs[usage][1],weight='bold',size=12,ha='right',va='bottom')

#     else:
#         sp.plot(x, eval('{} + {}*x'.format(
#                     traces['alpha'][:,col].mean(), traces['beta'][:,col].mean()))
#                     ,linewidth=2, color=clrs[usage][0], alpha=0.8)
    
#         sp.annotate('{}\nalpha: {:.2f}\nbeta:  {:.2f}'.format(usage
#                 ,traces['alpha'][:,col].mean(),traces['beta'][:,col].mean())
#                 ,xy=(0,1),xycoords='axes fraction',xytext=(12,-6),textcoords='offset points'
#                 ,color=clrs[usage][1],weight='bold',size=12,ha='left',va='top')        

       
        








# <headingcell level=2>

# Test Type

# <markdowncell>

# ### Unpooled

# <codecell>

## unpooled model for test type

unqvals_testtype = np.unique(df.testtype)
traces_unpooled = {}

for testtype in unqvals_testtype:
    x = df.loc[df.testtype == testtype, 'staard_score_pre']
    y = df.loc[df.testtype == testtype, 'staard_score_post']
    traces_unpooled[testtype] = get_traces_unpooled(x, y, max_iter=10000)

# <codecell>

### view parameters
for testtype in unqvals:
    print('Estimates for: {}'.format(testtype))
    pm.traceplot(traces_unpooled[testtype],figsize=(18,2*3))

# <markdowncell>

# ### Hierarchical

# <codecell>

## run traces
unqvals_testtype = np.unique(df.testtype)
unqvals_translator = {v:k for k,v in enumerate(unqvals_testtype)}
idxs = [unqvals_translator[v] for v in df.testtype]
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

fig, axes1d = plt.subplots(nrows=1, ncols=len(unqvals_testtype), sharex=True, sharey=True, figsize=(18,6))
fig.subplots_adjust(wspace=0.2)
fig.suptitle('Correlation of pre-test vs post-test scores - Bayesian Hierarchical Regression')
cm_cmap = cm.get_cmap('Set1')
    
for j, (sp, testype) in enumerate(zip(axes1d,unqvals_testtype)):

    print('{}: {}'.format(j,testype))
    clr = cm_cmap(j/len(unqvals_testtype))
    x = df.loc[df.testtype == testtype, 'staard_score_pre']
    y = df.loc[df.testtype == testtype, 'staard_score_post']
    
    # datapoints and subplot titles
    sp.scatter(x=x,y=y,s=40,color=clr,alpha=0.7,edgecolor='#333333')
    sp.annotate('{} ({} samples)'.format(testtype,len(x))
                ,xy=(0.5,1),xycoords='axes fraction',size=14,ha='center'
                ,xytext=(0,6),textcoords='offset points')

    plot_reg(sp,traces_unpooled[testtype],x,burn_ratio=0.25, ipoints=10)
#    plot_reg(sp,traces_hierarchical,x,usage='hierarchical',col=j,burn_ratio=0.5, ipoints=10)    
    

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

