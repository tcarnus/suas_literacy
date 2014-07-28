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
#     + [Outlier Removal](#Outlier-Removal)
# + [Frequentist Regression](#Frequentist-Regression)  
# + [Bayesian Regression](#Bayesian-Regression)  
#     + [Entire Set](#Entire-Set)  
#     + [Test Type](#Test-Type)  
#     + [Gender](#Gender)
#     + [Binned Age](#Binned-Age)
#     + [Binned PreScore](#Binned-PreScore)

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

from collections import OrderedDict

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

def get_traces_individual(x, y, max_iter=10000,quad=False):
    """ sample individual model """
    
    with pm.Model() as individual_model:

        # priors for alpha, beta and model error, uninformative  
        alpha = pm.Normal('alpha', mu=0, sd=100**2)
        beta = pm.Normal('beta', mu=0, sd=100**2)
        epsilon = pm.Uniform('epsilon', lower=0, upper=100)
        
        # configure model
        y_est = alpha + beta * x
        
        if quad:
            gamma = pm.Normal('gamma', mu=0, sd=100**2)
            y_est = alpha + beta * x + gamma * x ** 2

        # calc likelihood and do sampling
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



def plot_reg_bayes(df, xy, traces_ind, traces_hier, feat='no_feat', burn_ind=2000, burn_hier=None, quad=False, clr_school=True):
    """ create plot for bayesian derived regression lines, no groups """
    
    keys = traces_ind.keys()         
    fig, axes1d = plt.subplots(nrows=1, ncols=len(keys), sharex=True, sharey=True, figsize=(8*len(keys),8))
    fig.suptitle('Bayesian hierarchical regression of pre-test vs post-test scores')
    cm_cmap = cm.get_cmap('Set2')

    clrs = {}
    clrs['ind'] = ['#00F5FF','#006266','#00585C']
    clrs['hier'] = ['#FF7538','#661f00','#572610']
    
    point_clrs = 'cm_cmap(grp.clr)' if clr_school else 'cm_cmap(j/len(keys))'
    
    if len(keys) == 1:
        axes1d = [axes1d]
        
    for j, (sp, key) in enumerate(zip(axes1d,keys)):
        
        # scatterplot datapoints and subplot count title
        if feat == 'no_feat':
            x = df[xy['x']]
            y = df[xy['y']]
            for grpkey, grp in df.groupby('schoolid'):
                sp.scatter(grp[xy['x']],grp[xy['y']],s=40,color=eval(point_clrs),label='{} ({})'.format(grpkey,len(grp))
                           ,alpha=0.7,edgecolor='#333333')

        if feat != 'no_feat':
            x = df.loc[df[feat] == key,xy['x']]
            y = df.loc[df[feat] == key,xy['y']]
            for grpkey, grp in df.loc[df[feat] == key].groupby('schoolid'):
                sp.scatter(grp[xy['x']],grp[xy['y']],s=40,color=eval(point_clrs),label='{} ({})'.format(grpkey,len(grp))
                           ,alpha=0.7,edgecolor='#333333')
            
        sp.annotate('{} ({} samples)'.format(key,len(x))
            ,xy=(0.5,1),xycoords='axes fraction',size=14,ha='center'
            ,xytext=(0,6),textcoords='offset points')

        if clr_school:
            sp.legend(scatterpoints=1, loc=8, ncol=1, bbox_to_anchor=(1.0, 0.35), fancybox=True, shadow=True)
        
        # setup xlims and plot 1:1 line # BODGED the xlims
        xfit = np.linspace(x.min(), x.max(), 10)
        sp.plot(np.array([65,135]),np.array([65,135]),linestyle='dashed',linewidth=0.5,color='#999999')
        
        # plot actual data mean
        sp.scatter(x.mean(),y.mean(),marker='+',s=500,color='#551A8B')
        
        # plot regression: individual
        alpha = traces_ind[key]['alpha'][burn_ind:]
        beta = traces_ind[key]['beta'][burn_ind:]
        yfit = alpha[:, None] + beta[:, None] * xfit   # <- yfit for all samples at x in xfit ind
        yfit_at_xmean = alpha[:, None] + beta[:, None] * x.mean()
        note = '{}\nslope:  {:.2f}\nincpt: {:.2f}\niamx:  {:.2f}'.format('individual'
                            ,beta.mean(),alpha.mean(),yfit_at_xmean.mean()-x.mean())
        
        if quad:
            gamma = traces_ind[key]['gamma'][burn_ind:]
            yfit = alpha[:, None] + beta[:, None] * xfit + gamma[:, None] * xfit ** 2
            note = '{}\ny={:.2f} + {:.2f}x + {:.2f}x^2'.format('individual',alpha.mean(),beta.mean(),gamma.mean())
        
        mu = yfit.mean(0)
        yerr_975 = np.percentile(yfit,97.5,axis=0)
        yerr_025 = np.percentile(yfit,2.5,axis=0)
        
        sp.plot(xfit, mu,linewidth=3, color=clrs['ind'][0], alpha=0.8)
        sp.fill_between(xfit, yerr_025, yerr_975, color=clrs['ind'][2],alpha=0.3)
        sp.annotate(note,xy=(1,0),xycoords='axes fraction',xytext=(-12,6),textcoords='offset points'
                ,color=clrs['ind'][1],weight='bold',size=12,ha='right',va='bottom')

        # plot regression: hierarchical
        if traces_hier is not None:    
            alpha = traces_hier['alpha'][burn_hier:,j]
            beta = traces_hier['beta'][burn_hier:,j]
            yfit = alpha[:, None] + beta[:, None] * xfit
            yfit_at_xmean = alpha[:, None] + beta[:, None] * x.mean()
            note = '{}\nslope:  {:.2f}\nincpt: {:.2f}\niamx:  {:.2f}'.format('hierarchical'
                                ,beta.mean(),alpha.mean(),yfit_at_xmean.mean()-x.mean())
            
#             if quad:
#                 gamma = traces_hier['gamma'][burn_hier:,j]
#                 yfit = alpha[:, None] + beta[:, None] * xfit + gamma[:, None] * xfit ** 2
#                 note = '{}\ny={:.2f} + {:.2f}x + {:.2f}x^2'.format('individual',alpha.mean(),beta.mean(),gamma.mean())
            
            mu = yfit.mean(0)
            yerr_975 = np.percentile(yfit,97.5,axis=0)
            yerr_025 = np.percentile(yfit,2.5,axis=0)

            sp.plot(xfit, mu,linewidth=3, color=clrs['hier'][0], alpha=0.8)
            sp.fill_between(xfit, yerr_025, yerr_975, color=clrs['hier'][2], alpha=0.3)
            sp.annotate(note, xy=(0,1),xycoords='axes fraction',xytext=(100,-6),textcoords='offset points'
                ,color=clrs['hier'][1],weight='bold',size=12,ha='right',va='top')
        
    plt.show()

# <headingcell level=2>

# Import Data

# <codecell>

## Read cleaned dataset back from db
cnx_sql3 = sqlite3.connect('data/SUAS_data_master_v001_tcc_cleaned.db')
cnx_sql3.text_factory = str
dfa = pd.read_sql('select * from df_piv', cnx_sql3, index_col=['code','schoolid','gender','test','testtype']
                 , parse_dates=['date_pre','date_post'])
cnx_sql3.close()

dfa = dfa.sort_index(axis=1)
dfa.reset_index(inplace=True)
dfa.set_index('code',inplace=True)

print(dfa.shape)
dfa.head()

# <headingcell level=2>

# Outlier Removal

# <codecell>

## View pre scores
dfa.boxplot(column='staard_score_pre',by='testtype',sym='k+',vert=False
                            ,widths=0.8,notch=True,bootstrap=1000,figsize=[12,3.5])

# <markdowncell>

# **Observe**
# + There's a single sample in `paired_reading_wrat` that has a prescore > 140.
# + This outlier ought to be removed from the modelling since it's going to adversely affect linear regression models

# <codecell>

df = dfa.loc[dfa.staard_score_pre < 140].copy()
df.shape

# <markdowncell>

# ### Add color feature to school

# <codecell>

df_schoolkey = df.groupby('schoolid').size().reset_index()
df_schoolkey['clr'] = np.arange(0,1,1/len(df_schoolkey))
df_schoolkey.drop(0,inplace=True)
df = pd.merge(df, df_schoolkey, how ='left',left_on='schoolid', right_on='schoolid')
df.shape

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

# <codecell>

## What's the range of deltas?

df['staard_score_delta'] = df['staard_score_post'] - df['staard_score_pre']

df.boxplot(column='staard_score_delta',by='testtype',sym='k+',vert=False
                            ,widths=0.8,notch=True,bootstrap=1000,figsize=[12,3.5])

# <markdowncell>

# ---

# <headingcell level=1>

# Bayesian Regression

# <codecell>

xy={'x':'staard_score_pre', 'y':'staard_score_post'}

# <headingcell level=2>

# Entire Set

# <markdowncell>

# **(Run as individual)**

# <codecell>

## run sampling
traces_ind_all = OrderedDict()
traces_ind_all['all'] = get_traces_individual(df[xy['x']], df[xy['y']], max_iter=10000)

# <codecell>

## view parameters
p = pm.traceplot(traces_ind_all['all'],figsize=(18,1.5*3))
plt.show(p)

# <markdowncell>

# ### Plot regression

# <codecell>

plot_reg_bayes(df,xy,traces_ind_all,None,burn_ind=2000)

# <markdowncell>

# **Try Quadratic**

# <codecell>

## run sampling
traces_ind_all_quad = OrderedDict()
traces_ind_all_quad['all'] = get_traces_individual(df[xy['x']], df[xy['y']], max_iter=10000, quad=True)

# <codecell>

## view parameters
p = pm.traceplot(traces_ind_all_quad['all'],figsize=(18,1.5*4))
plt.show(p)

# <codecell>

plot_reg_bayes(df,xy,traces_ind_all_quad,None,burn_ind=2000, quad=True)

# <headingcell level=2>

# Test Type

# <markdowncell>

# ### Unpooled

# <codecell>

## run sampling
unqvals_testtype = np.unique(df.testtype)
traces_ind_testtype = OrderedDict()

for testtype in unqvals_testtype:
    x = df.loc[df.testtype == testtype, xy['x']]
    y = df.loc[df.testtype == testtype, xy['y']]
    traces_ind_testtype[testtype] = get_traces_individual(x, y, max_iter=10000)

# <codecell>

## view parameters
for testtype in unqvals_testtype:
    print('Estimates for: {}'.format(testtype))
    pm.traceplot(traces_ind_testtype[testtype],figsize=(18,1.5*3))

# <markdowncell>

# ### Hierarchical

# <codecell>

## run sampling
unqvals_translator = {v:k for k,v in enumerate(unqvals_testtype)}
idxs = [unqvals_translator[v] for v in df.testtype]
traces_hier_testtype = get_traces_hierarchical(df[xy['x']], df[xy['y']], idxs, max_iter=100000)

# <codecell>

## view parameters
with pm.Model() as hierarchical_model:
    pm.traceplot(traces_hier_testtype,figsize=(16,1.5*7))

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

## run sampling
unqvals_gender = np.unique(df.gender)
traces_ind_gender = OrderedDict()
for gender in unqvals_gender:
    x = df.loc[df.gender == gender, 'staard_score_pre']
    y = df.loc[df.gender == gender, 'staard_score_post']
    traces_ind_gender[gender] = get_traces_individual(x, y, max_iter=10000)  
    

# <codecell>

## view parameters
for gender in unqvals_gender:
    print('Estimates for: {}'.format(gender))
    pm.traceplot(traces_ind_gender[gender],figsize=(18,1.5*3))

# <markdowncell>

# ### Hierarchical

# <codecell>

## run sampling
unqvals_translator = {v:k for k,v in enumerate(unqvals_gender)}
idxs = [unqvals_translator[v] for v in df.gender]
traces_hier_gender = get_traces_hierarchical(df[xy['x']], df[xy['y']], idxs, max_iter=100000)

# <codecell>

## view parameters
with pm.Model() as hierarchical_model:
    pm.traceplot(traces_hier_gender,figsize=(18,1.5*7))

# <markdowncell>

# ### Plot comparison of hierarchical vs unpooled

# <codecell>

plot_reg_bayes(df, xy ,traces_ind_gender, traces_hier_gender
               , feat='gender', burn_ind=2000, burn_hier=50000)

# <headingcell level=2>

# Binned Age

# <markdowncell>

# Bin age at pre-score into 1-yearly sub groups

# <codecell>

# Bin age
df['binned_age_pre'] = df['age_pre'].apply(lambda x: round(x,0))

# <markdowncell>

# ### Unpooled

# <codecell>

## run sampling
unqvals_binned_age = np.unique(df.binned_age_pre)
traces_ind_binned_age = OrderedDict()
for binned_age in unqvals_binned_age:
    x = df.loc[df.binned_age_pre == binned_age, 'staard_score_pre']
    y = df.loc[df.binned_age_pre == binned_age, 'staard_score_post']
    traces_ind_binned_age[binned_age] = get_traces_individual(x, y, max_iter=10000)  
    

# <codecell>

## view parameters
for binned_age in unqvals_binned_age:
    print('Estimates for: {}'.format(binned_age))
    pm.traceplot(traces_ind_binned_age[binned_age],figsize=(18,1.5*3))

# <markdowncell>

# ### Hierarchical

# <codecell>

## run sampling
unqvals_translator = {v:k for k,v in enumerate(unqvals_binned_age)}
idxs = [unqvals_translator[v] for v in df.binned_age_pre]
traces_hier_binned_age = get_traces_hierarchical(df[xy['x']], df[xy['y']], idxs, max_iter=100000)

# <codecell>

## view parameters
with pm.Model() as hierarchical_model:
    pm.traceplot(traces_hier_binned_age,figsize=(18,1.5*7))

# <markdowncell>

# ### Plot comparison of hierarchical vs unpooled

# <codecell>

plot_reg_bayes(df, xy ,traces_ind_binned_age, traces_hier_binned_age
               , feat='binned_age_pre', burn_ind=2000, burn_hier=50000)

# <headingcell level=2>

# Binned PreScore

# <markdowncell>

# Bin pre-score into sub groups [0,90), [90, 110), [110,inf)

# <codecell>

# Bin prescore
df.loc[df.staard_score_pre < 90,'binned_staard_score_pre'] = 's < 90'
df.loc[(df.staard_score_pre >= 90) & (df.staard_score_pre < 110),'binned_staard_score_pre'] = '90 <= s < 110'
df.loc[df.staard_score_pre >= 110,'binned_staard_score_pre'] = 's >= 110'

# <markdowncell>

# ### Unpooled

# <codecell>

## run samples
unqvals_binned_prescore = ['s < 90', '90 <= s < 110', 's >= 110']
# unqvals_binned_prescore = ['s < 110', 's >= 110']
traces_ind_binned_prescore = OrderedDict()
for binned_prescore in unqvals_binned_prescore:
    x = df.loc[df.binned_staard_score_pre == binned_prescore, 'staard_score_pre']
    y = df.loc[df.binned_staard_score_pre == binned_prescore, 'staard_score_post']
    traces_ind_binned_prescore[binned_prescore] = get_traces_individual(x, y, max_iter=10000)  
    

# <codecell>

## view parameters
for binned_prescore in unqvals_binned_prescore:
    print('Estimates for: {}'.format(binned_prescore))
    pm.traceplot(traces_ind_binned_prescore[binned_prescore],figsize=(18,1.5*3))

# <markdowncell>

# ### Hierarchical

# <codecell>

## run sampling
unqvals_translator = {v:k for k,v in enumerate(unqvals_binned_prescore)}
idxs = [unqvals_translator[v] for v in df.binned_staard_score_pre]
traces_hier_binned_prescore = get_traces_hierarchical(df[xy['x']], df[xy['y']], idxs, max_iter=100000)

# <codecell>

## view parameters
with pm.Model() as hierarchical_model:
    pm.traceplot(traces_hier_binned_prescore,figsize=(18,1.5*7))

# <markdowncell>

# ### Plot comparison of hierarchical vs unpooled

# <codecell>

plot_reg_bayes(df, xy ,traces_ind_binned_prescore, traces_hier_binned_prescore
               , feat='binned_staard_score_pre', burn_ind=2000, burn_hier=50000)

# <markdowncell>

# ### Write data post-regression to SQL DB for temporary storage

# <codecell>

## write to local sqlite file
cnx_sql3 = sqlite3.connect('data/SUAS_data_master_v001_tcc_cleaned.db')
df['date_pre'] = df['date_pre'].apply(str)
df['date_post'] = df['date_post'].apply(str)
df.to_sql('df_post_reg',cnx_sql3,if_exists='replace')
cnx_sql3.close()

# <markdowncell>

# ---
# 
# _copyleft_ **Open Data Science Ireland Meetup Group 2014**  
# <a href='http://www.meetup.com/opendatascienceireland/'>ODSI@meetup.com</a>

