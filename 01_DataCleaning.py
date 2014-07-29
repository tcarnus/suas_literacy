# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# <div style="float:right"><img src='files/assets/img/Suas_Logo_Header.png'></img></div>
# 
# ## Suas Educational Development - Literacy Intervention Investigation
# 
# # 01 Data Cleaning
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
# + [Import Data](#Import-Data)
#     + [Remove and Clean Features](#Remove-and-Clean-Features)

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
for feat in ['gender','preorpost', 'test','testtype','test_color']:
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

## rename date1 to date
df.rename(columns={'date1':'date'},inplace=True)

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

## who has 3 instances of a test ??
dfr.loc[ntests[ntests > 2].index]

# <codecell>

## just for now, remove the third entry for these people with 3x reading tests

df2s = dfr.loc[ntests[ntests == 2].index].copy()
df3s = dfr.loc[ntests[ntests == 3].index].copy()
df3s = df3s.loc[((df3s.preorpost=='post') & (df3s.test_color=='blue')) == False]

# stack back together    
df_clean = pd.concat([df2s,df3s],axis=0)
df_clean.shape

# <codecell>

### Have we taken care of all NaNs?
for feat in df_clean.columns.values:
    print('{}: {}: {}'.format(feat,df_clean.loc[pd.isnull(df_clean[feat])].shape[0],df_clean[feat].dtype))

# <codecell>

## remove test_color and ind, they're just cluttering the dataframe
df_clean.drop(['test_color','ind'],axis=1,inplace=True)

# <codecell>

## reshape the dataframe to person-oriented
df_clean.reset_index(inplace=True)

# <markdowncell>

# #### Correct GroupWRAT

# <codecell>

df_clean.loc[df_clean.testtype=='group_wrat','testtype'] = 'acceleread_accelewrite'

# <headingcell level=2>

# Reshape dataframe

# <codecell>

## reshape the dataframe to person-oriented
df_clean.set_index(['code','schoolid','gender','test','testtype','preorpost'],inplace=True)
df_piv = df_clean.unstack(['preorpost'])

## flatten the multicolumn indexes and rename
df_piv.columns = ['_'.join(col).strip() for col in df_piv.columns.values]

## reapply correct datatypes
for feat in ['age_pre','age_post','raw_score_pre','raw_score_post','staard_score_pre','staard_score_post']:
    df_piv[feat] = df_piv[feat].astype(np.float64)

print(df_piv.shape)
df_piv.head()

# <markdowncell>

# ### Write cleaned data to SQL DB for temporary storage

# <codecell>

## write to local sqlite file
cnx_sql3 = sqlite3.connect('data/SUAS_data_master_v001_tcc_cleaned.db')
#cnx_sql3.text_factory = str
df_piv['date_pre'] = df_piv['date_pre'].apply(str)
df_piv['date_post'] = df_piv['date_post'].apply(str)
df_piv.to_sql('df_piv',cnx_sql3,if_exists='replace')
cnx_sql3.close()

# <markdowncell>

# ---

# <markdowncell>

# ---
# 
# _copyleft_ **Open Data Science Ireland Meetup Group 2014**  
# <a href='http://www.meetup.com/opendatascienceireland/'>ODSI@meetup.com</a>

