# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# <div style="float:right"><img src='files/assets/img/Suas_Logo_Header.png'></img></div>
# 
# ## Suas Educational Development - Literacy Intervention Investigation
# 
# # 01 Data Cleaning and Initial Exploration
# 
# Contact: Adelaide Nic Chartaigh [adelaide@suas.ie]("mailto:adelaide@suas.ie")  
# Author: Jonathan Sedar [jon.sedar@applied.ai]("mailto:jon.sedar@applied.ai")  
# Date: Spring / Summer 2014
# 
# This set of notebooks explore, statistically analyse and model the data for the Literacy Intervention Investigation via Open Data Science Ireland (ODSI).
# 
# * The notebooks are 'reproducable research': embedding code and writeup into a single document that can be executed and interrogated interactively.
# 
# * The summaries presented in these reports do not contain individually identifying information.
# 
# 
# #Contents
# 
# + [Setup](#Setup)  
# 
# 
# + [Import Data](#Import-Data)
#     + [Remove and Clean Features](#Remove-and-Clean-Features)
#     
#     
# + [Quick View of Dataset](#Quick-View-of-Dataset)

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

## Import raw data and quick look

raw = pd.read_excel('data/SUAS_data_master_v001_tcc.xlsx',sheetname=1)
raw.rename(columns=lambda s: s.translate(remove_punct_map), inplace=True)
raw.rename(columns=lambda x: '_'.join(x.lower().split()), inplace=True)

raw.head()

# <headingcell level=2>

# Remove and Clean Features

# <codecell>

df = raw.copy()

## Remove dupe columns
df.drop(['id','age_at_test','date','reading_age'],axis=1,inplace=True)

# <markdowncell>

# ### Ordinals: lowercasing, null handling

# <codecell>

# simple lowercasing and null removal
for feat in ['gender','preorpost', 'test','test_color']:
    df[feat] = df[feat].apply(lambda x: x.lower().strip() if pd.isnull(x) == False else "unknown")
    print('\n{}\n'.format(df.groupby(feat).size()))

# <markdowncell>

# ### Ordinals: custom value alignment

# <codecell>

## gender
df.loc[df.gender == 'male','gender'] = 'm'
df.loc[df.gender == 'female','gender'] = 'f'
df.groupby('gender').size()

# <markdowncell>

# ### Floats: rounding

# <codecell>

## age
df['age'] = df['age'].apply(lambda x : round(x,2))

# <markdowncell>

# ### Clean up dtypes

# <codecell>

### what's left with NaNs?
for feat in df.columns.values:
    print('{}: {}: {}'.format(feat,df.loc[pd.isnull(df[feat])].shape[0],df[feat].dtype))

# <codecell>

## drop percentile and dob for now, they're missing too much
df.drop(['dob','percentile'],axis=1,inplace=True)

# <codecell>

## drop entries without a score
df.dropna(inplace=True)

# <markdowncell>

# ### Remove codes where there is not both a pre and post score

# <codecell>

## set index, group by and count for 2 tests, remove all other samples
df.set_index(['code','test'], inplace=True)
df.sortlevel(inplace=True)
ntests = df.groupby(df.index).size()

# <codecell>

## remove rows where only 1 pre or post
for tup in ntests[ntests == 1].index:
    dfr = df.drop(tup)

# <codecell>

ntests.describe()

# <codecell>

## who has 3 ??
for tup in ntests[ntests > 2].index:
    print(dfr.loc[tup])

# <codecell>


# <codecell>

## just for now, remove the third entry

# stack the 2's
df2s = dfr.copy()
df2s.drop(df2s.index,inplace=True)
for tup in ntests[ntests == 2].index:
    df2s = pd.concat([df2s,dfr.loc[tup]],axis=0)

# stack the 3's    
df3s = dfr.copy()
df3s.drop(df3s.index,inplace=True)
for tup in ntests[ntests == 3].index:
    dfrr = dfr.loc[tup].copy()
    df3s = pd.concat([df3s,dfrr.loc[((dfrr.preorpost=='post') & (dfrr.test_color=='blue')) == False]],axis=0)

# stack back together    
df_clean = pd.concat([df2s,df3s],axis=0)    

# <markdowncell>

# ### Write cleaned data to SQL DB for temporary storage

# <codecell>

## write to local sqlite file
cnx_sql3 = sqlite3.connect('data/SUAS_data_master_v001_tcc_cleaned.db')
cnx_sql3.text_factory = str
df_clean['date1'] = df_clean['date1'].apply(str)
df_clean.to_sql('df_clean',cnx_sql3,if_exists='replace')
cnx_sql3.close()

# <markdowncell>

# ---

# <headingcell level=1>

# Quick View of Dataset

# <codecell>

## Read cleaned dataset back from db
cnx_sql3 = sqlite3.connect('data/SUAS_data_master_v001_tcc_cleaned.db')
cnx_sql3.text_factory = str
dfc = pd.read_sql('select * from df_clean', cnx_sql3, index_col='code', parse_dates='date1')
cnx_sql3.close()

print(dfc.shape)
dfc.head()

# <codecell>


# <codecell>


# <markdowncell>

# ---
# 
# _copyleft_ **Open Data Science Ireland Meetup Group 2014**  
# <a href='http://www.meetup.com/opendatascienceireland/'>ODSI@meetup.com</a>

