# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:03:15 2023

@author: emily.lynn
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# define functions
# 'props' is going to be some style formatting
# try/except keeps it from throwing an error on a text column
def style_negative(v, props=''):
    """style negative values in dataframe"""
    try:
        return props if v < 0 else None
    except:
        pass
    
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try:
        return props if v > 0 else None
    except:
        pass
    
def audience_simple(country):
    """"Show top countries"""
    if country == 'US':
        return 'US'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'

# load data
@st.cache_data # has streamlit load the data once and doesn't do it every time the page reloads
def load_data():
    df_agg = pd.read_csv('data/Aggregated_Metrics_By_Video.csv').iloc[1:, :]
    df_agg_sub = pd.read_csv('data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('data/All_Comments_Final.csv')
    df_time = pd.read_csv('data/Video_Performance_Over_Time.csv')
    
    # Add/adjust some new columns
    df_agg.columns = ['Video', 'Video title', 'Video publish time', 'Comments added', 'Shares', 'Dislikes', 'Likes', \
                      'Subscribers lost', 'Subscribers gained', 'RPM (USD)', 'CPM (USD)', 'Average % viewed', \
                      'Average view duration', 'Views', 'Watch time (hours)', 'Subscribers', 'Your estimated revenue (USD)', \
                      'Impressions', 'Impressions click-through rate (%)']
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'])
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
    df_agg['Avg duration sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    df_agg['Engagement ratio'] = (df_agg['Comments added'] + df_agg['Shares'] + df_agg['Dislikes'] + df_agg['Likes']) / df_agg.Views
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg.sort_values('Video publish time', ascending = False, inplace=True)
    df_time['Date'] = pd.to_datetime(df_time['Date'])
    
    return df_agg, df_agg_sub, df_comments, df_time

df_agg, df_agg_sub, df_comments, df_time = load_data()

# engineer data
df_agg_diff = df_agg.copy()
metric_data_12mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months = 12)
median_agg = df_agg_diff[df_agg_diff['Video publish time'] >= metric_data_12mo].median() # automatically only applies to numeric cols
numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
df_agg_diff.iloc[:,numeric_cols] = (df_agg_diff.iloc[:,numeric_cols] - median_agg).div(median_agg)

# build dashboard
add_sidebar = st.sidebar.selectbox('Aggregate or Individual Video', ('Aggregate Metrics', 'Individual Video Analysis'))

# Total picture
if add_sidebar == 'Aggregate Metrics':
    df_agg_metrics = df_agg[['Video publish time', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 'RPM (USD)', \
                             'Average % viewed', 'Avg duration sec', 'Engagement ratio', 'Views / sub gained']]
    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months = 6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months = 12)
    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].median()
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].median()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    count = 0
    for i in metric_medians6mo.index:
        with columns[count]:
            delta = (metric_medians6mo[i] - metric_medians12mo[i])/metric_medians12mo[i]
            st.metric(label = i, value = round(metric_medians6mo[i], 1), delta = "{:.2%}".format(delta))
            count += 1
            if count >= 5: # makes it so there are 5 metrics in each row
                count = 0
                
    df_agg_diff['Publish date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())
    df_agg_diff_final = df_agg_diff.loc[:, ['Video title', 'Publish date', 'Views', 'Likes', 'Subscribers', 'Avg duration sec', 'Engagement ratio', 'Views / sub gained']]
    
    # See pandas table styling guide
    df_agg_numeric_lst = df_agg_diff_final.median().index.tolist()
    df_to_pct = {}
    for i in df_agg_numeric_lst:
        df_to_pct[i] = '{:.1%}'.format
    st.dataframe(df_agg_diff_final.style.applymap(style_negative, props = 'color:red').applymap(style_positive, props = 'color:green').format(df_to_pct))
    

if add_sidebar == 'Individual Video Analysis':
    videos = tuple(df_agg['Video title'])
    video_select = st.selectbox('Pick A Video:', videos)
    
    # Get only the data for the video selected
    agg_filtered = df_agg[df_agg['Video title'] == video_select]
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
    agg_sub_filtered.sort_values('Is Subscribed', inplace=True) # Makes so true is always on top of the graph
    
    fig = px.bar(agg_sub_filtered, x='Views', y='Is Subscribed', color='Country', orientation='h')
    st.plotly_chart(fig) # has compatibility with most Python viz tools
    
    
    
    
    
    
    
    