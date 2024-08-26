#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary modules

# In[1]:


from base64 import urlsafe_b64decode
import sqlite3
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import time
import seaborn as sns
import sqlalchemy
import squarify
import logging
import time
from sqlalchemy import text 


# ## Connecting to DB

# In[2]:


from configparser import ConfigParser

import urllib3
config = ConfigParser()
config.read('config.ini')
conn = psycopg2.connect(host=config.get('prod', 'host'), dbname=config.get('prod', 'pg_db'), user=config.get('prod', 'pg_admin'), password=config.get('prod', 'pg_pass'), port = config.get('prod', 'pg_port'))
cur = conn.cursor()


# # Functions for R,F,M Generation

# In[3]:


def gen_recency(num_days,key,table,r_attr,):
    today = time.time() #Starting date to fetch data from
    prev = today - (86400*num_days) # Number of days converted to epoch time to get only those data of those days only
    if r_attr != None:
        query = f"""SELECT did, eventtime, key,segment->'{r_attr}'
                FROM   events_{table}
                WHERE key = '{key}' AND (eventtime <= {today} AND eventtime >= {prev});"""
    else:
        query = f"""SELECT did, eventtime, key,segment
                FROM   events_{table}
                WHERE key = '{key}' AND (eventtime <= {today} AND eventtime >= {prev});"""
    print(query)
    recency = pd.read_sql(query, conn)
    return recency

# In[4]:


def gen_frequency(num_days,key,table,f_attr):
    today = time.time() #Starting date to fetch data from
    prev = today - (86400*num_days) # Number of days converted to epoch time to get only those data of those days only
    if f_attr != None:
        query = f"""SELECT did, eventtime, key,segment->'{f_attr}'
                FROM   events_{table}
                WHERE key = '{key}' AND (eventtime <= {today} AND eventtime >= {prev});"""
    else:
        query = f"""SELECT did, eventtime, key,segment
                FROM   events_{table}
                WHERE key = '{key}' AND (eventtime <= {today} AND eventtime >= {prev});"""
    frequency = pd.read_sql(query, conn)
    return frequency

# In[5]:


def gen_monetary(num_days,key,table,m_attr):
    today = time.time() #Starting date to fetch data from
    prev = today - (86400*num_days) # Number of days converted to epoch time to get only those data of those days only
    if m_attr != None:
        query = f"""SELECT did, eventtime, key,segment->'{m_attr}'
                FROM   events_{table}
                WHERE key = '{key}' AND (eventtime <= {today} AND eventtime >= {prev});"""
    else:
        query = f"""SELECT did, eventtime, key,segment
                FROM   events_{table}
                WHERE key = '{key}' AND (eventtime <= {today} AND eventtime >= {prev});"""    
    monetary = pd.read_sql(query, conn)
    return monetary

# # Functions for two types of running formats [similar r,f and multiple r,f events]

# In[6]:


def rfm_sim(recency,monetary):
    today = time.time()
    recency['todays_epoch'] = pd.datetime.now()
    recency['recency'] = (recency['todays_epoch'] - recency['max_date']).dt.days
    r_labels, f_labels, m_labels = range(5, 0, -1), range(1, 6), range(1, 6) # Creating labels from 1 to 5
    recency['r_score'] = pd.qcut(recency['recency'].rank(
        method='first'), q=5, labels=r_labels, duplicates='drop').astype(int) # Giving a score of 1 to 5 for recency
    recency['f_score'] = pd.qcut(recency['frequency'].rank(
        method='first'), q=5, labels=f_labels).astype(int) # Giving a score of 1 to 5 for frequency
    monetary['m_score'] = pd.qcut(monetary['moetary'].rank(
        method='first'), q=5, labels=m_labels).astype(int) # Giving a score of 1 to 5 for monetary
    
    fin = pd.merge(recency,monetary,on='did',how='left')
    
    fin = fin.fillna(0)
    
    fin['rfm_sum'] = fin['r_score'] + fin['f_score'] + fin['m_score']# Sum all scores into one, eg: 8 (3 + 4 + 1)
    
    fin['rfm_label'] = ''
    
    fin = assign_label(fin, (5, 5), (4, 5), 'champions') #Champions - bought recently, buy often and spend the most
    fin = assign_label(fin, (3, 4), (4, 5), 'loyal customers')  # Loyal Customers - spend good money and often, responsive to promotions
    fin = assign_label(fin, (4, 5), (2, 3), 'potential loyalist') # Potential Loyalist - recent customers, but spent a good amount and bought more than once
    fin = assign_label(fin, (5, 5), (1, 1), 'new customers') # New Customers - bought most recently, but not often
    fin = assign_label(fin, (4, 4), (1, 1), 'promising') # Promising - recent shoppers, but haven’t spent much
    fin = assign_label(fin, (3, 3), (3, 3), 'needing attention') # Needing Attention - above average recency, frequency and monetary values; may not have bought very recently though
    fin = assign_label(fin, (3, 3), (1, 2), 'about to sleep') # About To Sleep - below average recency, frequency and monetary values; will lose them if not reactivated
    fin = assign_label(fin, (1, 2), (3, 4), 'at risk') # At Risk - spent big money and purchased often but long time ago; need to bring them back
    fin = assign_label(fin, (1, 2), (5, 5), 'cant loose them') # Can't Loose Them - made biggest purchases, and often but haven’t returned for a long time
    fin = assign_label(fin, (1, 2), (1, 2), 'hibernating')
    
    fin = assign_label(fin, (1, 2), (0,0), 'hibernating')
    fin = assign_label(fin, (5, 5), (0,0), 'new customers')
    fin = assign_label(fin, (3, 3), (0,0), 'about to sleep')
    fin = assign_label(fin, (4, 4), (0, 0), 'promising') 

    return fin


# In[7]:


def rfm_dis(recency,frequency,moetary):
    today = time.time()
    recency['todays_epoch'] = pd.datetime.now()
    recency['recency'] = (recency['todays_epoch'] - recency['max_date']).dt.days
    r_labels, f_labels, m_labels = range(5, 0, -1), range(1, 6), range(1, 6) # Creating labels from 1 to 5
    recency['r_score'] = pd.qcut(recency['recency'].rank(
        method='first'), q=5, labels=r_labels, duplicates='drop').astype(int) # Giving a score of 1 to 5 for recency
    frequency['f_score'] = pd.qcut(frequency['frequency'].rank(
        method='first'), q=5, labels=f_labels).astype(int) # Giving a score of 1 to 5 for frequency
    moetary['m_score'] = pd.qcut(moetary['moetary'].rank(
        method='first'), q=5, labels=m_labels).astype(int) # Giving a score of 1 to 5 for monetary
    
    lev_1 = pd.merge(recency,moetary,on='did',how='left')
    fin = pd.merge(lev_1,frequency,on='did',how='left')
    
    fin = fin.fillna(0)
    
    fin['rfm_sum'] = fin['r_score'] + fin['f_score'] + fin['m_score']# Sum all scores into one, eg: 8 (3 + 4 + 1)
    
    fin['rfm_label'] = ''
    
    fin = assign_label(fin, (5, 5), (4, 5), 'champions') #Champions - bought recently, buy often and spend the most
    fin = assign_label(fin, (3, 4), (4, 5), 'loyal customers')  # Loyal Customers - spend good money and often, responsive to promotions
    fin = assign_label(fin, (4, 5), (2, 3), 'potential loyalist') # Potential Loyalist - recent customers, but spent a good amount and bought more than once
    fin = assign_label(fin, (5, 5), (1, 1), 'new customers') # New Customers - bought most recently, but not often
    fin = assign_label(fin, (4, 4), (1, 1), 'promising') # Promising - recent shoppers, but haven’t spent much
    fin = assign_label(fin, (3, 3), (3, 3), 'needing attention') # Needing Attention - above average recency, frequency and monetary values; may not have bought very recently though
    fin = assign_label(fin, (3, 3), (1, 2), 'about to sleep') # About To Sleep - below average recency, frequency and monetary values; will lose them if not reactivated
    fin = assign_label(fin, (1, 2), (3, 4), 'at risk') # At Risk - spent big money and purchased often but long time ago; need to bring them back
    fin = assign_label(fin, (1, 2), (5, 5), 'cant loose them') # Can't Loose Them - made biggest purchases, and often but haven’t returned for a long time
    fin = assign_label(fin, (1, 2), (1, 2), 'hibernating')
    
    fin = assign_label(fin, (1, 2), (0,0), 'hibernating')
    fin = assign_label(fin, (5, 5), (0,0), 'new customers')
    fin = assign_label(fin, (3, 3), (0,0), 'about to sleep')
    fin = assign_label(fin, (4, 4), (0, 0), 'promising') 
    

    return fin


# ## Function to Assign labels to the users

# In[8]:


def assign_label(df, r_rule, fm_rule, label, colname='rfm_label'):
    df.loc[(df['r_score'].between(r_rule[0], r_rule[1])) # labeling the groups according to the range provided
           & (df['f_score'].between(fm_rule[0], fm_rule[1])), colname] = label
    return df


# # Run Function

# In[9]:

def run(table_key,r_key,f_key,m_key,m_attribute=None,r_attribute=None,f_attribute=None,time_range=30):
    recency_key = r_key
    frequency_key = f_key
    table = table_key
    time = time_range

    #r_attr= r_attribute 
    r_attr = 'test' if r_attribute == None else r_attribute
    f_attr = 'test' if f_attribute == None else f_attribute
    m_attr = 'test' if m_attribute == None else m_attribute
    monetary_key = m_key  # Discuss tomorrow...INVEST_NOW_Page
    insert_query = f"""INSERT INTO rfmid (rkey, rattribute, fkey, fattribute, mkey, mattribute)
                    VALUES (
                     'Session_Start',
                     'test',
                     'Session_Start',
                     'test',
                     'MultiAccount',
                     'test') RETURNING *;"""
    #insert = pd.read-sql(insert_query,conn)
    insert =pd.io.sql.read_sql(insert_query,conn)
    #insert =sqlalchemy.read_sql(insert_query,conn)
    #insert=pd.read_sql(text("SELECT * from rfmid"),conn)

    def urllib3 = ('postgresql+psycopg2://username:Pradeep@123@host:port/database')
engine = sqlalchemy.create_engine(urlsafe_b64decode)
#with engine.connect().execution_options(autocommit=True) as conn:
    #df = pd.read_sql(f"""SELECT * FROM rfmid""", con = conn) 
    #from sqlalchemy import create_engine
#engine = create_engine(urllib3.connection_from_url) 
#conn = psycopg2.connect(host=config.get('prod', 'host'), dbname=config.get('prod', 'pg_db'), user=config.get('prod', 'pg_admin'), password=config.get('prod', 'pg_pass'), port = config.get('prod', 'pg_port'))
#cur = conn.cursor()
with engine.connect().execution_options(autocommit=True) as conn:
    query = conn.execute(text(sql))         
df = pd.DataFrame(query.fetchall())
    
     

    
    if recency_key == frequency_key:
        # calling recency function
        recency = gen_recency(time, recency_key, table ,r_attr)
        # calling moetary function
        monetary = gen_monetary(time, monetary_key, table , m_attr)
        # Converting the epoch time to pd Timestamp
        recency['eventtime'] = pd.to_datetime(recency['eventtime'], unit='s')

        agg_dict1 = {
            'eventtime': 'max',
            'key': 'count'}

        agg_dict2 = {
            'key': 'count'}

        recency = recency.groupby('did').agg(agg_dict1).reset_index()
        monetary = monetary.groupby('did').agg(agg_dict2) .reset_index()

        recency.columns = ['did', 'max_date', 'frequency']
        monetary.columns = ['did', 'monetary']
        r = recency
        m = monetary
        for i in range(len(recency['did'])):
           #upsert_Summary_query = f"""INSERT INTO RFMSummary.Summary 
                                       #VALUES (date = {datetime.now()},
                                             # rfm_value = {rfm_value},
                                              #count )"""
            upsert_Detail_query = f"""INSERT INTO RFMDetails.Detail 
                                       VALUES (did = {recency[did[i]]},
                                           prev_rfm_score = current_rfm_score ,
                                           current_rfm_score = {recency[max_date[i]]} ,
                                           prev_rfm_date = current_rfm_date , 
                                           current_rfm_date = {datetime.now()}) 
                                           ON CONFLICT did 
                                           UPDATE SET prev_rfm_score = current_rfm_score ,
                                                  current_rfm_value = {rfm_scores} ,
                                                  prev_rfm_date = current_rfm_date , 
                                                  current_rfm_date = {datetime.now()};"""
            upsert_Detail = pd.read_sql(upsert_Detail_query,conn)
            #upsert_Summary = pd.read_sql(upsert_Summary_query,conn)
        #return rfm_sim(recency, monetary)
        return insert 



    else:
        # calling recency function
        recency = gen_recency(time, recency_key, table, r_attr)
        # calling frequency function
        frequency = gen_frequency(time, frequency_key, table, f_attr)
        # calling moetary function
        monetary = gen_monetary(time, monetary_key, table, m_attr)
        # Converting the epoch time to pd Timestamp
        recency['eventtime'] = pd.to_datetime(recency['eventtime'], unit='s')

        agg_dict1 = {
            'eventtime': 'max'}
        agg_dict2 = {
            'key': 'count'}

        recency = recency.groupby('did').agg(agg_dict1).reset_index()
        frequency = frequency.groupby('did').agg(agg_dict2).reset_index()
        monetary = monetary.groupby('did').agg(agg_dict2).reset_index()

        recency.columns = ['did', 'max_date']
        frequency.columns = ['did', 'frequency']
        monetary.columns = ['did', 'monetary']
        for i in range (len(recency)):
            #upsert_Summary_query = f"""INSERT INTO RFMSummary.Summary 
                                      # VALUES (date = {datetime.now()},
                                             # rfm_value = {rfm_value},
                                              #count )"""
            upsert_Detail_query = f"""INSERT INTO RFMDetails.Detail 
                                       VALUES (did = {rencency[did[i]]},
                                           prev_rfm_score = current_rfm_score ,
                                           current_rfm_score = {rfm_scores} ,
                                           prev_rfm_date = current_rfm_date , 
                                           current_rfm_date = {datetime.now()})
                                           ON CONFLICT did 
                                           UPDATE SET prev_rfm_score = current_rfm_score ,
                                                  current_rfm_value = {rfm_scores} ,
                                                  prev_rfm_date = current_rfm_date , 
                                                  current_rfm_date = {datetime.now()};"""
            upsert_Detail = pd.read_sql(upsert_Detail_query,conn)
            #upsert_Summary = pd.read_sql(upsert_Summary_query,conn)
        return rfm_dis(recency, frequency, monetary)


# In[10]:


df = run('5bebe93c25d705690ffbc758', 'Session_Start' ,'Session_Start','MultiAccount') 


# ### Visualization

# In[11]:


agg_dict2 = {
    'did': 'count',
    'recency': 'mean',
    'frequency': 'mean',
    'moetary': 'sum'
}

#df_analysis = df.groupby('rfm_label').agg(agg_dict2).sort_values(by='recency').reset_index()
#df_analysis.rename({'rfm_label': 'label', 'did': 'count'}, axis=1, inplace=True)
#df_analysis['count_share'] = df_analysis['count'] / df_analysis['count'].sum()
#df_analysis['moetary_share'] = df_analysis['moetary'] / df_analysis['moetary'].sum()
#df_analysis['moetary'] = df_analysis['moetary'] / df_analysis['count']


# In[58]:


# if non-division error
#df_analysis['moetary'] = df_analysis['moetary'].add(1)
#df_analysis['count'] = df_analysis['count'].add(1)


# In[12]:


#df_analysis


# In[13]:


colors = ['#37BEB0', '#DBF5F0', '#41729F', '#C3E0E5', '#0C6170', '#5885AF', '#E1C340', '#274472', '#F8EA8C', '#A4E5E0', '#1848A0']

#for col in ['count', 'moetary']:
    #labels = df_analysis['label'] + df_analysis[col+'_share'].apply(lambda x: ' ({0:.1f}%)'.format(x*100))

    #fig, ax = plt.subplots(figsize=(16,6))
    #squarify.plot(sizes= df_analysis[col], label=labels, alpha=.8, color=colors)
    #ax.set_title('RFM Segments of Customers (%s)' % col)
    #plt.axis('off')
    #plt.show()

