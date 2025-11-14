import time
import streamlit as st

import pandas as pd
from pandas import json_normalize
import requests


from io import BytesIO
import numpy as np

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import datetime

from scipy import stats

url = st.secrets["API_URL"]

response = requests.get(url)
data = response.json()

df = pd.DataFrame(data['経過'])

parameters = ['月齢', '前後径', '左右径', '頭囲', '短頭率', '前頭部対称率', 'CA', '後頭部対称率', 'CVAI', 'CI']
df[parameters] = df[parameters].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.sort_values('月齢')

df_h = pd.DataFrame(data['ヘルメット'])
df_h = df_h[(df_h['ダミーID'] != '') & (df_h['ヘルメット'] != '')]

df_c = pd.DataFrame(data['患者数'])
df_c['診察日'] = pd.to_datetime(df_c['診察日'], format='mixed')
df_c = df_c.sort_values('診察日')
df_c['患者総数'] = range(1, len(df_c) + 1)
df_c['治療患者総数'] = ((df_c['発注有無'] == '発注済').astype(int)).cumsum()
df_c['治療患者総数'] = df_c['治療患者総数'].where((df_c['発注有無'] == '発注済'), other=None).ffill()

fig = go.Figure()

# df_fig = df_fig[df_fig['診察日'] < '2025-10-01']
df_fig = df_c.copy()

fig.add_trace(go.Histogram(x = df_fig[df_fig['発注有無'] == '発注済']['診察日'], marker=dict(color='blue', opacity=0.75), name = '1月あたりの初診患者数（治療あり）', yaxis='y1'))
fig.add_trace(go.Histogram(x = df_fig[df_fig['発注有無'] != '発注済']['診察日'], marker=dict(color='cyan', opacity=0.75), name = '1月あたりの初診患者数（治療なし）', yaxis='y1'))
fig.add_trace(go.Scatter(x = df_fig['診察日'],  y = df_fig['治療患者総数'], mode = 'lines', marker=dict(color='blue'), name = '治療患者総数', yaxis='y2'))
fig.add_trace(go.Scatter(x = df_fig['診察日'],  y = df_fig['患者総数'], mode = 'lines', marker=dict(color='cyan'), name = '患者総数', yaxis='y2'))

fig.add_vline(x="2023-02-11", line_width=2, line_dash="dash", line_color="grey")  #関西院
fig.add_vline(x="2024-03-17", line_width=2, line_dash="dash", line_color="grey")  #表参道院
fig.add_vline(x="2025-03-14", line_width=2, line_dash="dash", line_color="grey")  #福岡院

# レイアウトの指定
fig.update_layout(height=900,width = 1600,  #16:9に、
                  plot_bgcolor='white', #背景色を白に
                  title_text='患者数の推移',
                  xaxis = dict(type='date', dtick = 'M1'), # dtick: 1か月ごとは'M1'
                  yaxis = dict(title = '人数（1月あたり）', side = 'left', showgrid=False, # ２軸だと見誤る場合があるので目盛り線は表示させない(showgrid=False)
                               range = [0, 520]),    # rangeで指定したほうがよい。ゼロが合わない場合などがある。
                  yaxis2 = dict(title = 'のべ患者数', side = 'right', overlaying = 'y', range = [0, max(df_fig['患者総数'])], showgrid=False),
                  bargap = 0.2,
                  barmode = 'stack',
                  legend=dict(yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01),
                  font_size=20
                  )

# st.plotly_chart(fig)

treated_patients = df_h['ダミーID'].unique()
df_first = df[df['治療ステータス'] == '治療前'].drop_duplicates('ダミーID')

df_tx = df[df['ダミーID'].isin(treated_patients)]
df_tx_pre_last = df_tx[df_tx['治療ステータス'] == '治療前'].drop_duplicates('ダミーID', keep='last')

df_tx_pre_last['治療前月齢'] = df_tx_pre_last['月齢']

category_orders={'治療前PSRレベル':['レベル1', 'レベル2', 'レベル3', 'レベル4'],
                   '治療前ASRレベル':['レベル1', 'レベル2', 'レベル3', 'レベル4'],
                   '治療前短頭症':['重症', '中等症', '軽症', '正常', '長頭'],
                   '治療前CA重症度':['正常', '軽症', '中等症', '重症', '最重症'],
                   '治療前CVAI重症度':['正常', '軽症', '中等症', '重症', '最重症'],
                   '治療前の月齢':[i for i in range(15)],
                   '初診時の月齢':[i for i in range(15)]}

def add_pre_levels(df):
  df['治療前PSRレベル'] = ''
  df['治療前PSRレベル'] = df['治療前PSRレベル'].mask(df['後頭部対称率']>=90, 'レベル1')
  df['治療前PSRレベル'] = df['治療前PSRレベル'].mask(df['後頭部対称率']<90, 'レベル2')
  df['治療前PSRレベル'] = df['治療前PSRレベル'].mask(df['後頭部対称率']<85, 'レベル3')
  df['治療前PSRレベル'] = df['治療前PSRレベル'].mask(df['後頭部対称率']<80, 'レベル4')

  df['治療前ASRレベル'] = ''
  df['治療前ASRレベル'] = df['治療前ASRレベル'].mask(df['前頭部対称率']>=90, 'レベル1')
  df['治療前ASRレベル'] = df['治療前ASRレベル'].mask(df['前頭部対称率']<90, 'レベル2')
  df['治療前ASRレベル'] = df['治療前ASRレベル'].mask(df['前頭部対称率']<85, 'レベル3')
  df['治療前ASRレベル'] = df['治療前ASRレベル'].mask(df['前頭部対称率']<80, 'レベル4')

  df['治療前CA重症度'] = '正常'
  df['治療前CA重症度'] = df['治療前CA重症度'].mask(df['CA']>6, '軽症')
  df['治療前CA重症度'] = df['治療前CA重症度'].mask(df['CA']>9, '中等症')
  df['治療前CA重症度'] = df['治療前CA重症度'].mask(df['CA']>13, '重症')
  df['治療前CA重症度'] = df['治療前CA重症度'].mask(df['CA']>17, '最重症')

  df['治療前CVAI重症度'] = '正常'
  df['治療前CVAI重症度'] = df['治療前CVAI重症度'].mask(df['CVAI']>5, '軽症')
  df['治療前CVAI重症度'] = df['治療前CVAI重症度'].mask(df['CVAI']>7, '中等症')
  df['治療前CVAI重症度'] = df['治療前CVAI重症度'].mask(df['CVAI']>10, '重症')
  df['治療前CVAI重症度'] = df['治療前CVAI重症度'].mask(df['CVAI']>14, '最重症')

  df['治療前短頭症'] = ''
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']>126, '長頭')
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']<=126, '正常')
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']<106, '軽症')
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']<103, '中等症')
  df['治療前短頭症'] = df['治療前短頭症'].mask(df['短頭率']<100, '重症')

  return(df)

def add_post_levels(df):
  df['最終PSRレベル'] = ''
  df['最終PSRレベル'] = df['最終PSRレベル'].mask(df['後頭部対称率']>=90, 'レベル1')
  df['最終PSRレベル'] = df['最終PSRレベル'].mask(df['後頭部対称率']<90, 'レベル2')
  df['最終PSRレベル'] = df['最終PSRレベル'].mask(df['後頭部対称率']<85, 'レベル3')
  df['最終PSRレベル'] = df['最終PSRレベル'].mask(df['後頭部対称率']<80, 'レベル4')

  df['最終ASRレベル'] = ''
  df['最終ASRレベル'] = df['最終ASRレベル'].mask(df['前頭部対称率']>=90, 'レベル1')
  df['最終ASRレベル'] = df['最終ASRレベル'].mask(df['前頭部対称率']<90, 'レベル2')
  df['最終ASRレベル'] = df['最終ASRレベル'].mask(df['前頭部対称率']<85, 'レベル3')
  df['最終ASRレベル'] = df['最終ASRレベル'].mask(df['前頭部対称率']<80, 'レベル4')

  df['最終CA重症度'] = '正常'
  df['最終CA重症度'] = df['最終CA重症度'].mask(df['CA']>6, '軽症')
  df['最終CA重症度'] = df['最終CA重症度'].mask(df['CA']>9, '中等症')
  df['最終CA重症度'] = df['最終CA重症度'].mask(df['CA']>13, '重症')
  df['最終CA重症度'] = df['最終CA重症度'].mask(df['CA']>17, '最重症')

  df['最終CVAI重症度'] = '正常'
  df['最終CVAI重症度'] = df['最終CVAI重症度'].mask(df['CVAI']>5, '軽症')
  df['最終CVAI重症度'] = df['最終CVAI重症度'].mask(df['CVAI']>7, '中等症')
  df['最終CVAI重症度'] = df['最終CVAI重症度'].mask(df['CVAI']>10, '重症')
  df['最終CVAI重症度'] = df['最終CVAI重症度'].mask(df['CVAI']>14, '最重症')

  df['最終短頭症'] = ''
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']>126, '長頭')
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']<=126, '正常')
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']<106, '軽症')
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']<103, '中等症')
  df['最終短頭症'] = df['最終短頭症'].mask(df['短頭率']<100, '重症')

  return(df)

df_tx_pre_last = add_pre_levels(df_tx_pre_last)

#経過も利用する場合
df_tx_post =  df_tx[df_tx['治療ステータス'] == '治療後']

df_tx_pre_age = df_tx_pre_last[['ダミーID', '月齢']]
df_tx_pre_age = df_tx_pre_age.rename(columns = {'月齢':'治療前月齢'})

df_tx_post = pd.merge(df_tx_post, df_tx_pre_age, on='ダミーID', how='left')

df_tx_post['治療期間'] = df_tx_post['月齢'] - df_tx_post['治療前月齢']
df_period = df_tx_post[['ダミーID', '治療期間']]

df_tx_pre_last['治療期間'] = 0

#df_tx_post = pd.merge(df_tx_post, df_tx_pre_last[['ダミーID']+list(category_orders.keys())], on='ダミーID', how='left')
df_tx_post = pd.merge(df_tx_post, df_tx_pre_last[['ダミーID','治療前PSRレベル', '治療前ASRレベル', '治療前短頭症', '治療前CA重症度', '治療前CVAI重症度']], on='ダミーID', how='left')

df_tx_pre_post = pd.concat([df_tx_pre_last, df_tx_post])

df_tx_pre_post = pd.merge(df_tx_pre_post, df_h, on='ダミーID', how='left')

df_tx_post_last = df_tx_post.drop_duplicates('ダミーID', keep='last')

df_tx_post_last = add_post_levels(df_tx_post_last)

df_tx_pre_post = pd.merge(df_tx_pre_post, df_tx_post_last[['ダミーID','最終PSRレベル', '最終ASRレベル', '最終短頭症', '最終CA重症度', '最終CVAI重症度']], on='ダミーID', how='left')

#経過観察
df_first = add_pre_levels(df_first)
#df_pre_age = df_first[['ダミーID', '月齢']+list(category_orders.keys())]
df_pre_age = df_first[['ダミーID', '月齢', '治療前PSRレベル', '治療前ASRレベル', '治療前短頭症', '治療前CA重症度', '治療前CVAI重症度']]
df_pre_age = df_pre_age.rename(columns = {'月齢':'治療前月齢'})

df_co = pd.merge(df, df_pre_age, on='ダミーID', how='left')
df_co = df_co[df_co['治療ステータス'] == '治療前']
obs_patients = df_co[df_co['ダミーID'].duplicated()]['ダミーID'].unique()
df_co = df_co[df_co['ダミーID'].isin(obs_patients)]

# IDごとに最大と最小の年齢を計算
age_diff_df = df_co.groupby('ダミーID')['月齢'].agg(['max', 'min']).reset_index()

# 年齢差を新しいカラムとして追加
age_diff_df['治療期間'] = age_diff_df['max'] - age_diff_df['min']

df_co = pd.merge(df_co, age_diff_df[['ダミーID', '治療期間']], on='ダミーID', how='left')

df_co['ヘルメット'] = '経過観察'
#df_co['治療ステータス'] = df_co['治療ステータス'].mask(~df_co['ダミーID'].duplicated(), '治療後')
df_co['治療ステータス'] = df_co.groupby('ダミーID')['月齢'].transform(lambda x: ['治療前'] + ['治療後'] * (len(x) - 1))
df_co['ダミーID'] = df_co['ダミーID'] + 'C'

df_tx_pre_post = pd.concat([df_tx_pre_post, df_co])

df_tx_pre_post['治療前の月齢'] = df_tx_pre_post['治療前月齢'].apply(lambda x: np.floor(x) if pd.notnull(x) else np.nan)

df_co['治療前の月齢'] = df_co['治療前月齢'].apply(lambda x: np.floor(x) if pd.notnull(x) else np.nan)

df_co = add_post_levels(df_co)

# Streamlitアプリのページ設定
st.set_page_config(page_title='位置的頭蓋変形に関するデータの可視化', page_icon="📊", layout='wide')

#治療率ありでパラメータごとにヒストグラムを作成（go.Barを利用）
def hist(parameter='短頭率', df_first=df_first):
  import plotly.graph_objects as go

  all_number = len(df_first['ダミーID'].unique())

  df_first[parameter] = pd.to_numeric(df_first[parameter], errors='coerce')
  df_first[parameter] = df_first[parameter].round()

  df_first_tx = df_first[df_first['ダミーID'].isin(treated_patients)]
  tx_number = len(df_first_tx['ダミーID'].unique())
  tx_rate = round((tx_number/all_number)*100, 1)

  treated = []
  untreated = []
  all = []
  tx_rates=[]

  min = int(df_first[parameter].min())
  max_para = int(df_first[parameter].max())

  for i in list(range(min, max_para)):
    tx_n = df_first_tx[df_first_tx[parameter] == i][parameter].count()
    all_n = df_first[df_first[parameter] == i][parameter].count()
    untx_n = all_n-tx_n
    if all_n > 0:
      rate = (tx_n/all_n)*100
      rate = round(rate, 1)
    else:
      rate = ''

    treated.append(round(tx_n, 1))
    untreated.append(round(untx_n, 1))
    all.append(round(all_n, 1)) #不要？
    tx_rates.append(rate)

  x=list(range(min, max_para))

  y=[0, max(all)]

  fig = go.Figure(go.Bar(x=x, y=treated, name='治療あり', marker_color='blue')) #opacity=0.8
  fig.add_trace(go.Bar(x=x, y=untreated, name='治療なし',  marker_color='cyan', text=tx_rates)) #opacity=0.4
  fig.update_traces(textfont_size=12, textfont_color='black',
                    #textangle=0,
                    textposition="outside", cliponaxis=False)

  if parameter == '短頭率':
    limits=list({106, 126} & set(x))
  elif parameter in ['前頭部対称率', '後頭部対称率']:
    limits=list({80, 85, 90} & set(x))
  elif parameter == 'CA':
    limits=list({6, 9, 13, 17} & set(x))
  elif parameter == 'CVAI':
    limits=list({5, 7, 10, 14} & set(x))
  elif parameter == 'CI':
    limits=list({80, 94, 101} & set(x))

  for i in range(len(limits)):
    #fig.add_trace(go.Line(x=[limits[i],limits[i]], y=y, mode='lines', marker_color='pink', line=dict(dash='dot'), name=str(limits[i])))
    #fig.add_trace(go.scatter.Line(x=[limits[i],limits[i]], y=y, mode='lines', marker_color='pink', line=dict(dash='dot'), name=str(limits[i])))
    fig.add_trace(go.Scatter(
        x=[limits[i], limits[i]],  # x座標
        y=y,                       # y座標
        mode='lines',              # 線を描画
        marker_color='pink',
        line=dict(dash='dot'),
        name=str(limits[i])
    ))

  if all_number >= 1000:
    all_number = str(all_number)
    digits = len(all_number)
    all_number = all_number[:digits-3] + ',' + all_number[digits-3:]
  else:
    all_number = str(all_number)

  fig.update_layout(width=1600, height=900,
      plot_bgcolor='white',
      title_text=parameter+'の分布（全'+all_number+'人で'+str(tx_rate)+'％が治療）',
      xaxis_title_text=parameter,
      yaxis_title_text='人数',
      barmode='stack'
      )

  st.plotly_chart(fig)

def show_helmet_proportion():
  # 色をカスタマイズ
  colors = ['red', 'green', 'blue']

  # ヘルメットの種類ごとに行の数を集計
  counts = df_h['ヘルメット'].value_counts().reset_index()
  counts.columns = ['ヘルメット', '数']

  # 円グラフ作成
  fig = px.pie(counts, names='ヘルメット', values='数', color_discrete_sequence=colors)
  fig.update_layout(width=900, title='ヘルメットの種類の内訳')

  # Streamlitアプリにグラフを表示
  st.plotly_chart(fig)

def show_age_proportion(df):
  df_fig = df.copy()

  df_age = pd.DataFrame()
  
  df_young = df_fig[df_fig['治療前月齢'] < 3]
  df_young['治療前月齢'] = '-2'

  df_age = pd.concat([df_age, df_young])

  for i in range(3, 12):
    df_temp = df_fig[(df_fig['治療前月齢'] >= i) & (df_fig['治療前月齢'] < i+1)]
    df_temp['治療前月齢'] = str(i)
    df_age = pd.concat([df_age, df_temp])

  df_old = df_fig[df_fig['治療前月齢'] >= 12]
  df_old['治療前月齢'] = '12-'
  
  df_age = pd.concat([df_age, df_old])

  # 月齢の順序リストを定義
  age_order = ['-2'] + [str(i) for i in range(3, 12)] + ['12-']
  df_age['治療前月齢'] = pd.Categorical(df_age['治療前月齢'], categories=age_order, ordered=True)

  # カウント＆整列
  counts = df_age['治療前月齢'].value_counts().sort_index().reset_index()
  counts.columns = ['治療前月齢', '数']

  # 円グラフ作成
  fig = px.pie(counts, names='治療前月齢', values='数', category_orders={'治療前月齢': age_order})
  fig.update_layout(width=900, title='治療前月齢の割合')

  # Streamlitアプリにグラフを表示
  st.plotly_chart(fig)

def takamatsu(df, brachy=False):
  df_analysis = df.copy()
  df_analysis['ASR'] = df_analysis['前頭部対称率']
  df_analysis['PSR'] = df_analysis['後頭部対称率']
  df_analysis['BI'] = df_analysis['短頭率']

  ranges={'CA':[6, 9, 13, 17], 'CVAI':[5, 7, 10, 14], 'ASR':[90, 85, 80], 'PSR':[90, 85, 80], 'CI':[78, 95], 'BI':[126,106,103,100]}

  dftx_pre = df_analysis[df_analysis['治療ステータス'] == '治療前']

  parameters=['CA', 'CVAI', 'ASR', 'PSR', 'BI', 'CI']

  classifications = {'CA':['normal', 'mild', 'moderate', 'severe', 'very severe'], 'CVAI':['normal', 'mild', 'moderate', 'severe', 'very severe'],
                    'ASR':['Level1', 'Level2', 'Level3', 'Level4'], 'PSR':['Level1', 'Level2', 'Level3', 'Level4'],
                    'CI':['dolicho', 'meso', 'brachy'],
                    'BI':['dolicho', 'meso', 'mild', 'moderate', 'severe']}

  definitions = {'CA':['0-5', '6-8', '9-12', '13-16', '=>17'], 'CVAI':['0-4', '5-6', '7-9', '10-13', '=>14'],
                    'ASR':['>90', '86-90', '81-85', '=<80'], 'PSR':['>90', '86-90', '81-85', '=<80'],
                    'CI':['=<78', '79-94', '=>95'], 'BI':['>126', '106-126', '103-106', '100-103', '<100']}

  df_vis = pd.DataFrame()
  order=0

  for parameter in parameters:
    df_temp = dftx_pre[['ダミーID', parameter]]
    df_temp['指標'] = parameter
    df_temp['Classification'] = ''
    if parameter in ['CA', 'CVAI']:
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]<ranges[parameter][0], 'normal')
      for i in range(len(ranges[parameter])-1):
        df_temp['Classification'] = df_temp['Classification'].mask((df_temp[parameter]>=ranges[parameter][i])&(df_temp[parameter]<ranges[parameter][i+1]), classifications[parameter][i+1])
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]>=ranges[parameter][-1], 'very severe')

    elif parameter in ['ASR', 'PSR']:
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]>ranges[parameter][0], 'Level1')
      for i in range(len(ranges[parameter])-1):
        df_temp['Classification'] = df_temp['Classification'].mask((df_temp[parameter]<=ranges[parameter][i])&(df_temp[parameter]>ranges[parameter][i+1]), classifications[parameter][i+1])
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]<=ranges[parameter][-1], 'Level4')

    elif parameter == 'CI':
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]<=ranges[parameter][0], classifications[parameter][0])
      df_temp['Classification'] = df_temp['Classification'].mask((df_temp[parameter]<ranges[parameter][1])&(df_temp[parameter]>ranges[parameter][0]), classifications[parameter][1])
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]>=ranges[parameter][1], classifications[parameter][2])

    elif parameter == 'BI':
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]>ranges[parameter][0], classifications[parameter][0])
      for i in range(len(ranges[parameter])-1):
        df_temp['Classification'] = df_temp['Classification'].mask((df_temp[parameter]<=ranges[parameter][i])&(df_temp[parameter]>ranges[parameter][i+1]), classifications[parameter][i+1])
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]<=ranges[parameter][-1], classifications[parameter][-1])


    df_temp = df_temp.groupby(['指標', 'Classification']).count()[['ダミーID']] #.astype(int).astype(str)

    df_temp = df_temp.rename(columns={'ダミーID': 'Before Helmet'})
    df_temp['Before Helmet'] = df_temp['Before Helmet'].fillna(0).astype(int)
    df_temp['%']=round((df_temp['Before Helmet']/len(dftx_pre))*100, 1)
    df_temp['%']=df_temp['%'].astype(str)
    df_temp['%']='('+df_temp['%']+'%)'

    df_temp.loc[(parameter, 'average: '+parameter+' (SD)'), 'Before Helmet'] = round(dftx_pre[parameter].mean(), 2)
    sd = dftx_pre[parameter].std()
    df_temp.loc[(parameter, 'average: '+parameter+' (SD)'), '%'] = '(SD '+str(round(sd, 1))+')'

    df_vis = pd.concat([df_vis, df_temp])
    if order == 0:
      df_vis['Definition']=''
      df_vis['order']=''

    c=0
    for classification in classifications[parameter]:
      df_vis.loc[(parameter, classification), 'Definition'] = definitions[parameter][c]
      df_vis.loc[(parameter, classification), 'order'] = order
      #print(order)
      c += 1
      order += 1

    df_vis.loc[(parameter, 'average: '+parameter+' (SD)'), 'order'] = order
    order += 1

  df_vis_pre = df_vis.sort_values('order')
  df_vis_pre = df_vis_pre[['Definition', 'Before Helmet', '%']]
  df_vis_pre = df_vis_pre.fillna('')

  dftx_post = df_analysis.drop_duplicates('ダミーID', keep='last')

  df_vis = pd.DataFrame()
  order=0

  for parameter in parameters:
    #print(parameter)
    df_temp = dftx_post[['ダミーID', parameter]]
    df_temp['指標'] = parameter
    df_temp['Classification'] = ''
    if parameter in ['CA', 'CVAI']:
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]<ranges[parameter][0], 'normal')
      for i in range(len(ranges[parameter])-1):
        df_temp['Classification'] = df_temp['Classification'].mask((df_temp[parameter]>=ranges[parameter][i])&(df_temp[parameter]<ranges[parameter][i+1]), classifications[parameter][i+1])
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]>=ranges[parameter][-1], 'very severe')

    elif parameter in ['ASR', 'PSR']:
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]>ranges[parameter][0], 'Level1')
      for i in range(len(ranges[parameter])-1):
        df_temp['Classification'] = df_temp['Classification'].mask((df_temp[parameter]<=ranges[parameter][i])&(df_temp[parameter]>ranges[parameter][i+1]), classifications[parameter][i+1])
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]<=ranges[parameter][-1], 'Level4')

    elif parameter == 'CI':
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]<=ranges[parameter][0], classifications[parameter][0])
      df_temp['Classification'] = df_temp['Classification'].mask((df_temp[parameter]<ranges[parameter][1])&(df_temp[parameter]>ranges[parameter][0]), classifications[parameter][1])
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]>=ranges[parameter][1], classifications[parameter][2])

    elif parameter == 'BI':
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]>ranges[parameter][0], classifications[parameter][0])
      for i in range(len(ranges[parameter])-1):
        df_temp['Classification'] = df_temp['Classification'].mask((df_temp[parameter]<=ranges[parameter][i])&(df_temp[parameter]>ranges[parameter][i+1]), classifications[parameter][i+1])
      df_temp['Classification'] = df_temp['Classification'].mask(df_temp[parameter]<=ranges[parameter][-1], classifications[parameter][-1])

    df_temp = df_temp.groupby(['指標', 'Classification']).count()[['ダミーID']] #.astype(int).astype(str)

    df_temp = df_temp.rename(columns={'ダミーID': 'After Helmet'})
    df_temp['After Helmet'] = df_temp['After Helmet'].fillna(0).astype(int)
    #df_temp['After Helmet'] = df_temp['After Helmet'].astype(int)
    df_temp['%']=round((df_temp['After Helmet']/len(dftx_post))*100, 1)
    df_temp = df_temp.fillna(0)
    df_temp['%']=df_temp['%'].astype(str)
    df_temp['%']='('+df_temp['%']+'%)'

    df_temp.loc[(parameter, 'average: '+parameter+' (SD)'), 'After Helmet'] = round(dftx_post[parameter].mean(), 2)
    sd = dftx_post[parameter].std()
    df_temp.loc[(parameter, 'average: '+parameter+' (SD)'), '%'] = '(SD '+str(round(sd, 1))+')'

    df_vis = pd.concat([df_vis, df_temp])
    if order == 0:
      df_vis['order']=''

    for classification in classifications[parameter]:
      df_vis.loc[(parameter, classification), 'order'] = order
      #print(order)
      order += 1

    df_vis.loc[(parameter, 'average: '+parameter+' (SD)'), 'order'] = order
    order += 1

  df_vis_post = df_vis.sort_values('order')
  df_vis_post = df_vis_post.fillna(0)

  df_vis_post['%'] = df_vis_post['%'].mask(df_vis_post['%']==0, '( 0.0%)')
  df_vis_post = df_vis_post[['After Helmet', '%']]

  df_vis = pd.merge(df_vis_pre, df_vis_post, left_on=['指標', 'Classification'], right_index=True)
  df_vis = df_vis[['Definition', 'Before Helmet', '%_x', 'After Helmet', '%_y']]
  df_vis = df_vis.rename(columns={'%_x': '%', '%_y': '% '})

  #人数を整数に
  df_vis['Before Helmet'] = df_vis['Before Helmet'].mask(df_vis['Before Helmet']%1==0, df_vis['Before Helmet'].astype(int).astype(str))
  df_vis['After Helmet'] = df_vis['After Helmet'].mask(df_vis['After Helmet']%1==0, df_vis['After Helmet'].astype(int).astype(str))
  return(df_vis)

def graham(df, parameter, border=False, x_limit=False):
  fig = make_subplots(
      rows=1, cols=6,
      # 初めに各グラフのタイトルを設定
      subplot_titles=('-3', '4', '5', '6', '7', '8-'),
      shared_yaxes=True
  )

  df_fig = df.copy()

  df_age = pd.DataFrame()
  
  df_young = df_fig[df_fig['治療前月齢'] < 4]
  df_young['治療前月齢'] = '-3'

  df_age = pd.concat([df_age, df_young])

  for i in range(4, 8):
    df_temp = df_fig[(df_fig['治療前月齢'] >= i) & (df_fig['治療前月齢'] < i+1)]
    df_temp['治療前月齢'] = str(i)
    df_age = pd.concat([df_age, df_temp])

  df_old = df_fig[df_fig['治療前月齢'] >= 8]
  df_old['治療前月齢'] = '8-'
  
  df_age = pd.concat([df_age, df_old])

  df_fig = df_age.copy()

  df_pre = df_fig[df_fig['治療ステータス'] == '治療前']
  df_fig = df_fig.sort_values('月齢')  #不要？
  df_fig = df_fig.drop_duplicates('ダミーID', keep='last')

  severities = {'後頭部対称率':'治療前PSRレベル', '前頭部対称率':'治療前ASRレベル', 'CA':'治療前CA重症度', 'CVAI':'治療前CVAI重症度', '短頭率':'治療前短頭症', 'CI':'治療前短頭症'}
  severities = severities[parameter]

  parameter_names = {'後頭部対称率':'PSR', '前頭部対称率':'ASR', 'CA':'CA', 'CVAI':'CVAI', '短頭率':'BI', 'CI':'CI'}
  parameter_name = parameter_names[parameter]

  if parameter in ['後頭部対称率', '前頭部対称率']:
    levels = ['レベル1', 'レベル2', 'レベル3', 'レベル4']
  elif parameter in ['CA', 'CVAI']:
    levels = ['軽症', '中等症', '重症', '最重症']
  else:
    levels = ['軽症', '中等症', '重症']

  line_colors = ['blue', 'green', 'black', 'red', 'purple']
  #line_colors = ['rgb(150,150,150)', 'rgb(100,100,100)', 'rgb(50,50,50)', 'black']
  dashes = ['solid', 'dashdot', 'dash', 'dot'] #'longdash', 'longdashdot'

  import math
  ages = ['-3', '4', '5', '6', '7', '8-']

  #print('治療前月齢のリスト', ages)
  #st.write('治療前月齢のリスト:', ages)

  max_sd0, max_sd1 = 0, 0

  range_max = 0

  x_rage_mins = {}

  x_rage_maxes = {}

  for i, age in enumerate(ages, 1):
    if i > 6:  # 最大6列まで
      break
      
    df_temp = df_fig[df_fig['治療前月齢'] == age]
    #df_temp = df_fig[(df_fig['治療前月齢'] >= age) & (df_fig['治療前月齢'] < age+1)]
    df_pre_min = df_pre[df_pre['治療前月齢'] == age]
    #df_pre_min = df_pre[(df_pre['治療前月齢'] >= age) & (df_pre['治療前月齢'] < age+1)]

    #min = df_pre_min['月齢'].min()
    min = 20
    #max = df_temp['月齢'].max()
    max = 0

    x_rage_mins[age] = 20
    x_rage_maxes[age] = 0

    #for level, line_color in zip(levels, line_colors):
    for level, line_color, dash in zip(levels, line_colors, dashes):
      df_temp_temp = df_temp[df_temp[severities] == level]
      temp_members = df_temp_temp['ダミーID'].unique()
      df_pre_temp = df_pre[df_pre['ダミーID'].isin(temp_members)]

      x, x_sd, y, y_sd = [], [], [], []

      mean0 = df_pre_temp['月齢'].mean()
      x.append(mean0)

      mean1 = df_temp_temp['月齢'].mean()
      x.append(mean1)

      sd0 = df_pre_temp['月齢'].std()
      x_sd.append(sd0)

      if max_sd0 < sd0:
        max_sd0 = sd0

      if min > mean0 - sd0:
        min = mean0 - sd0*1.1

      sd = df_temp_temp['月齢'].std()
      x_sd.append(sd)

      if max_sd1 < sd:
        max_sd1 = sd

      if max < mean1 + sd:
         #max = mean1 + sd*1.1 + sd0*1.1
         max = mean1 + sd*1.1

      if x_rage_mins[age] > min:
        x_rage_mins[age] = min
      
      if x_rage_maxes[age] < max:
        x_rage_maxes[age] = max

      #月齢の幅
      range_age = max - min
      if range_max < range_age:
        range_max = range_age

      #y.append(df_pre_temp['治療前'+parameter].mean())
      y.append(df_pre_temp[parameter].mean())
      #y.append(df_temp_temp['最終'+parameter].mean())
      y.append(df_temp_temp[parameter].mean())
      #y_sd.append(df_pre_temp['治療前'+parameter].std())
      y_sd.append(df_pre_temp[parameter].std())
      #y_sd.append(df_temp_temp['最終'+parameter].std())
      y_sd.append(df_temp_temp[parameter].std())

      if i == 1:
        d = go.Scatter(x=x, y=y,
                    error_x=dict(type='data', array=x_sd, visible=True),
                    error_y=dict(type='data', array=y_sd, visible=True),
                    mode='markers+lines',
                    #line=dict(color = line_color),
                    line=dict(color = line_color, dash = dash),
                    #ids=[level, level],
                    #name=age + level
                    name = level,
                    legendgroup=age)
                    #legendgroup=level)
      else:
        d = go.Scatter(x=x, y=y,
                    error_x=dict(type='data', array=x_sd, visible=True),
                    error_y=dict(type='data', array=y_sd, visible=True),
                    mode='markers+lines',
                    #line=dict(color = line_color),
                    line=dict(color = line_color, dash = dash),
                    showlegend=False,  #ここが違う
                    #ids=[level, level],
                    #name=age + level
                    #name = level,
                    #legendgroup=age
                    )

      #print(fig.print_grid())  #グリッド構造を確認
      #fig.append_trace(d, 1, i)
      fig.add_trace(d, row=1, col=i)

    if border:
      if parameter == 'CVAI':
        upper_border = 6.25
        lower_border = 3.5
      elif parameter == 'CA':
        upper_border = 6
        lower_border = False
      elif parameter == 'CI':
        upper_border = 94
        lower_border = False
      else:
        upper_border = 90
        lower_border = False


      #CVAI = 6.25
      d = go.Scatter(mode='lines',
                    x=[0, 25],
                      y=[upper_border]*2,
                      line=dict(color = 'black', dash='dot'),
                      showlegend=False,
                      #name='CVAI=5%'
                      )
      #fig.append_trace(d, 1, i)
      fig.add_trace(d, row=1, col=i)

      if lower_border:
        #CVAI = 3.5
        d = go.Scatter(mode='lines',
                      x=[0, 25],
                        y=[lower_border]*2,
                        line=dict(color = 'black', dash='dash'),
                        showlegend=False,
                        #name='CVAI=3.5%'
                        )
        #fig.append_trace(d, 1, i)
        fig.add_trace(d, row=1, col=i)

  #print(range_max)

  #表示範囲の設定
  if parameter == 'CVAI':
    min, max = 0, 19

  elif parameter == 'CA':
    min, max = 0, 25

  elif parameter == '前頭部対称率':
    min, max = 70, 100
  
  elif parameter == '後頭部対称率':
    min, max = 60, 100

  elif parameter == '短頭率':
    min, max = 94, 114
  else:  #CI？
    min, max = 89, 109

  premargin = 0.5
  if max_sd0 > 0.5:
    premargin = max_sd0*1.1

  range_max = 0

  for age in ages:
    range_age = x_rage_maxes[age] - x_rage_mins[age]
    if range_max < range_age:
      range_max = range_age

  if x_limit:
    layout = go.Layout(width=1600, height=900,
                      title='Change in '+parameter_name+' on Age & Severity Groups',
                      #paper_bgcolor='white',
                      #xaxis=dict(title='age', range=[2-premargin, 1.5+range_max]), 
                      #xaxis=dict(title='age', range=[3, 3 + x_limit+1]),
                      xaxis=dict(title='age', range=[x_rage_mins['-3'], x_rage_mins['-3'] + x_limit+1]),
                      #xaxis2=dict(title='age', range=[4-premargin, 3.5+range_max]),
                      xaxis2=dict(title='age', range=[4, 4 + x_limit+1]),
                      #xaxis3=dict(title='age', range=[5-premargin, 4.5+range_max]),
                      xaxis3=dict(title='age', range=[5, 5 + x_limit+1]),
                      #xaxis4=dict(title='age', range=[6-premargin, 5.5+range_max]),
                      xaxis4=dict(title='age', range=[6, 6 + x_limit+1]),
                      #xaxis5=dict(title='age', range=[7-premargin, 6.5+range_max]),
                      xaxis5=dict(title='age', range=[7, 7 + x_limit+1]),
                      #xaxis6=dict(title='age', range=[8-premargin, 7.5+range_max]),
                      xaxis6=dict(title='age', range=[8, 8 + x_limit+1]),
                      yaxis=dict(title='Mean '+parameter_name, range=[min, max]),
                      yaxis2=dict(range=[min, max]),
                      yaxis3=dict(range=[min, max]),
                      yaxis4=dict(range=[min, max]),
                      yaxis5=dict(range=[min, max]),
                      yaxis6=dict(range=[min, max]))
  else:
    layout = go.Layout(width=1600, height=900,
                      title='Change in '+parameter_name+' on Age & Severity Groups',
                      #paper_bgcolor='white',
                      #xaxis=dict(title='age', range=[2-premargin, 1.5+range_max]), 
                      xaxis=dict(title='age', range=[x_rage_mins['-3'], x_rage_mins['-3'] + range_max]),
                      #xaxis2=dict(title='age', range=[4-premargin, 3.5+range_max]),
                      xaxis2=dict(title='age', range=[x_rage_mins['4'], x_rage_mins['4'] + range_max]),
                      #xaxis3=dict(title='age', range=[5-premargin, 4.5+range_max]),
                      xaxis3=dict(title='age', range=[x_rage_mins['5'], x_rage_mins['5'] + range_max]),
                      #xaxis4=dict(title='age', range=[6-premargin, 5.5+range_max]),
                      xaxis4=dict(title='age', range=[x_rage_mins['6'], x_rage_mins['6'] + range_max]),
                      #xaxis5=dict(title='age', range=[7-premargin, 6.5+range_max]),
                      xaxis5=dict(title='age', range=[x_rage_mins['7'], x_rage_mins['7'] + range_max]),
                      #xaxis6=dict(title='age', range=[8-premargin, 7.5+range_max]),
                      xaxis6=dict(title='age', range=[x_rage_mins['8-'], x_rage_mins['8-'] + range_max]),
                      yaxis=dict(title='Mean '+parameter_name, range=[min, max]),
                      yaxis2=dict(range=[min, max]),
                      yaxis3=dict(range=[min, max]),
                      yaxis4=dict(range=[min, max]),
                      yaxis5=dict(range=[min, max]),
                      yaxis6=dict(range=[min, max]))

  fig['layout'].update(layout)

  fig.update_layout(plot_bgcolor="white")
  fig.update_xaxes(linecolor='gray', linewidth=2)
  fig.update_yaxes(gridcolor='lightgray')

  #return(fig)
  #st.plotly_chart(fig)

  import uuid
  st.plotly_chart(fig, key=str(uuid.uuid4()))

def graham_hc(df, border=False, x_limit=False):
  fig = make_subplots(
      rows=1, cols=6,
      # 初めに各グラフのタイトルを設定
      subplot_titles=('-3', '4', '5', '6', '7', '8-'),
      shared_yaxes=True
  )

  df_fig = df.copy()

  df_age = pd.DataFrame()
  
  df_young = df_fig[df_fig['治療前月齢'] < 4]
  df_young['治療前月齢'] = '-3'

  df_age = pd.concat([df_age, df_young])

  for i in range(4, 8):
    df_temp = df_fig[(df_fig['治療前月齢'] >= i) & (df_fig['治療前月齢'] < i+1)]
    df_temp['治療前月齢'] = str(i)
    df_age = pd.concat([df_age, df_temp])

  df_old = df_fig[df_fig['治療前月齢'] >= 8]
  df_old['治療前月齢'] = '8-'
  
  df_age = pd.concat([df_age, df_old])

  df_fig = df_age.copy()

  df_pre = df_fig[df_fig['治療ステータス'] == '治療前']
  df_fig = df_fig.sort_values('月齢')  #不要？
  df_fig = df_fig.drop_duplicates('ダミーID', keep='last')

  line_colors = ['blue', 'green', 'black', 'red', 'purple']
  #line_colors = ['rgb(150,150,150)', 'rgb(100,100,100)', 'rgb(50,50,50)', 'black']
  dashes = ['solid', 'dashdot', 'dash', 'dot'] #'longdash', 'longdashdot'

  import math
  ages = ['-3', '4', '5', '6', '7', '8-']

  max_sd0, max_sd1 = 0, 0

  range_max = 0

  x_rage_mins = {}
  x_rage_maxes = {}

  for i, age in enumerate(ages, 1):
    if i > 6:  # 最大6列まで
      break
      
    df_temp = df_fig[df_fig['治療前月齢'] == age]
    #df_temp = df_fig[(df_fig['治療前月齢'] >= age) & (df_fig['治療前月齢'] < age+1)]
    df_pre_min = df_pre[df_pre['治療前月齢'] == age]
    #df_pre_min = df_pre[(df_pre['治療前月齢'] >= age) & (df_pre['治療前月齢'] < age+1)]

    #min = df_pre_min['月齢'].min()
    min = 20
    #max = df_temp['月齢'].max()
    max = 0

    x_rage_mins[age] = 20
    x_rage_maxes[age] = 0

    #for level, line_color, dash in zip(levels, line_colors, dashes):
    line_color = 'black'
    dash = 'solid'
    #df_temp_temp = df_temp[df_temp[severities] == level]
    temp_members = df_temp['ダミーID'].unique()
    df_pre_temp = df_pre[df_pre['ダミーID'].isin(temp_members)]

    x, x_sd, y, y_sd = [], [], [], []

    mean0 = df_pre_temp['月齢'].mean()
    x.append(mean0)

    mean1 = df_temp['月齢'].mean()
    x.append(mean1)

    sd0 = df_pre_temp['月齢'].std()
    x_sd.append(sd0)

    if max_sd0 < sd0:
      max_sd0 = sd0

    if min > mean0 - sd0:
      min = mean0 - sd0*1.1

    sd = df_temp['月齢'].std()
    x_sd.append(sd)

    if max_sd1 < sd:
      max_sd1 = sd

    if max < mean1 + sd:
        #max = mean1 + sd*1.1 + sd0*1.1
        max = mean1 + sd*1.1

    if x_rage_mins[age] > min:
      x_rage_mins[age] = min
    
    if x_rage_maxes[age] < max:
      x_rage_maxes[age] = max

    #月齢の幅
    range_age = max - min
    if range_max < range_age:
      range_max = range_age

    #y.append(df_pre_temp['治療前'+parameter].mean())
    y.append(df_pre_temp['頭囲'].mean())
    #y.append(df_temp_temp['最終'+parameter].mean())
    y.append(df_temp['頭囲'].mean())
    #y_sd.append(df_pre_temp['治療前'+parameter].std())
    y_sd.append(df_pre_temp['頭囲'].std())
    #y_sd.append(df_temp_temp['最終'+parameter].std())
    y_sd.append(df_temp['頭囲'].std())

    # if i == 1:
    #   d = go.Scatter(x=x, y=y,
    #               error_x=dict(type='data', array=x_sd, visible=True),
    #               error_y=dict(type='data', array=y_sd, visible=True),
    #               mode='markers+lines',
    #               #line=dict(color = line_color),
    #               line=dict(color = line_color, dash = dash),
    #               #ids=[level, level],
    #               #name=age + level
    #               #name = level,
    #               legendgroup=age)
    #               #legendgroup=level)
    # else:
    d = go.Scatter(x=x, y=y,
                error_x=dict(type='data', array=x_sd, visible=True),
                error_y=dict(type='data', array=y_sd, visible=True),
                mode='markers+lines',
                #line=dict(color = line_color),
                line=dict(color = line_color, dash = dash),
                showlegend=False,  #ここが違う
                #ids=[level, level],
                #name=age + level
                #name = level,
                #legendgroup=age
                )

    #print(fig.print_grid())  #グリッド構造を確認
    #fig.append_trace(d, 1, i)
    fig.add_trace(d, row=1, col=i)

  #print(range_max)

  #表示範囲の設定
  min, max = 380, 480

  premargin = 0.5
  if max_sd0 > 0.5:
    premargin = max_sd0*1.1

  range_max = 0

  for age in ages:
    range_age = x_rage_maxes[age] - x_rage_mins[age]
    if range_max < range_age:
      range_max = range_age

  if x_limit:
    layout = go.Layout(width=1600, height=900,
                      title='Change in head circumference on age groups',
                      #paper_bgcolor='white',
                      #xaxis=dict(title='age', range=[2-premargin, 1.5+range_max]), 
                      xaxis=dict(title='age', range=[3, 3 + x_limit+1]),
                      #xaxis2=dict(title='age', range=[4-premargin, 3.5+range_max]),
                      xaxis2=dict(title='age', range=[4, 4 + x_limit+1]),
                      #xaxis3=dict(title='age', range=[5-premargin, 4.5+range_max]),
                      xaxis3=dict(title='age', range=[5, 5 + x_limit+1]),
                      #xaxis4=dict(title='age', range=[6-premargin, 5.5+range_max]),
                      xaxis4=dict(title='age', range=[6, 6 + x_limit+1]),
                      #xaxis5=dict(title='age', range=[7-premargin, 6.5+range_max]),
                      xaxis5=dict(title='age', range=[7, 7 + x_limit+1]),
                      #xaxis6=dict(title='age', range=[8-premargin, 7.5+range_max]),
                      xaxis6=dict(title='age', range=[8, 8 + x_limit+1]),
                      #yaxis=dict(title='Mean '+parameter_name, range=[min, max]),
                      yaxis=dict(title='Mean head circumference', range=[min, max]), 
                      yaxis2=dict(range=[min, max]),
                      yaxis3=dict(range=[min, max]),
                      yaxis4=dict(range=[min, max]),
                      yaxis5=dict(range=[min, max]),
                      yaxis6=dict(range=[min, max]))
  else:
    layout = go.Layout(width=1600, height=900,
                      title='Change in head circumference on age groups',
                      #paper_bgcolor='white',
                      #xaxis=dict(title='age', range=[2-premargin, 1.5+range_max]), 
                      xaxis=dict(title='age', range=[x_rage_mins['-3'], x_rage_mins['-3'] + range_max]),
                      #xaxis2=dict(title='age', range=[4-premargin, 3.5+range_max]),
                      xaxis2=dict(title='age', range=[x_rage_mins['4'], x_rage_mins['4'] + range_max]),
                      #xaxis3=dict(title='age', range=[5-premargin, 4.5+range_max]),
                      xaxis3=dict(title='age', range=[x_rage_mins['5'], x_rage_mins['5'] + range_max]),
                      #xaxis4=dict(title='age', range=[6-premargin, 5.5+range_max]),
                      xaxis4=dict(title='age', range=[x_rage_mins['6'], x_rage_mins['6'] + range_max]),
                      #xaxis5=dict(title='age', range=[7-premargin, 6.5+range_max]),
                      xaxis5=dict(title='age', range=[x_rage_mins['7'], x_rage_mins['7'] + range_max]),
                      #xaxis6=dict(title='age', range=[8-premargin, 7.5+range_max]),
                      xaxis6=dict(title='age', range=[x_rage_mins['8-'], x_rage_mins['8-'] + range_max]),
                      #yaxis=dict(title='Mean '+parameter_name, range=[min, max]),
                      yaxis=dict(title='Mean head circumference', range=[min, max]),
                      yaxis2=dict(range=[min, max]),
                      yaxis3=dict(range=[min, max]),
                      yaxis4=dict(range=[min, max]),
                      yaxis5=dict(range=[min, max]),
                      yaxis6=dict(range=[min, max]))

  fig['layout'].update(layout)

  fig.update_layout(plot_bgcolor="white")
  fig.update_xaxes(linecolor='gray', linewidth=2)
  fig.update_yaxes(gridcolor='lightgray')

  #return(fig)
  import uuid
  st.plotly_chart(fig, key=str(uuid.uuid4()))

def animate_BI_PSR(df0, df):
  colors = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8C33', '#33FFF1', '#8C33FF', '#FF5733', '#57FF33', '#5733FF',
    '#FF3357', '#33FFA1', '#FFA133', '#33FF8C', '#FF338C', '#8CFF33', '#A1FF33', '#338CFF', '#A133FF', '#33A1FF'
  ]

  #df0 = df.drop_duplicates('ダミーID', keep='first')

  df1 = df.drop_duplicates('ダミーID', keep='last')

  common_patients = set(df1['ダミーID'].unique()) & (set(df0['ダミーID'].unique()))

  df = pd.concat([df0, df1])
  df = df[df['ダミーID'].isin(common_patients)]

  #複数のヘルメットを使用している患者を除外
  df_helmet = df[df['ヘルメット'] != '経過観察']
  helmet_counts = df_helmet.groupby('ダミーID')['ヘルメット'].nunique()
  common_patients = helmet_counts[helmet_counts > 1].index.tolist()

  df = df[~df['ダミーID'].isin(common_patients)]

  fig = px.scatter(df, x='短頭率', y='後頭部対称率', color='治療前PSRレベル', symbol='治療前短頭症', facet_col = 'ヘルメット',
                   hover_data=['ダミーID', '治療期間', '治療前月齢', 'ヘルメット'] + parameters, category_orders=category_orders, animation_frame='治療ステータス', animation_group='ダミーID', color_discrete_sequence=colors)
  i=0
  for i in range(len(df['ヘルメット'].unique())):
    #短頭率の正常範囲
    fig.add_trace(go.Scatter(x=[106, 106], y=[df['後頭部対称率'].min(), 100], mode='lines', line=dict(color='gray', dash = 'dot'), name='短頭率正常下限'), row=1, col=i+1)
    fig.add_trace(go.Scatter(x=[126, 126], y=[df['後頭部対称率'].min(), 100], mode='lines', line=dict(color='gray', dash = 'dot'), name='短頭率正常上限'), row=1, col=i+1)

    #対称率の正常範囲
    fig.add_trace(go.Scatter(x=[df['短頭率'].min(), df['短頭率'].max()], y=[90, 90], mode='lines', line=dict(color='gray', dash = 'dot'), name='後頭部対称率正常下限'), row=1, col=i+1)

  fig.update_xaxes(range = [df['短頭率'].min()-2,df['短頭率'].max()+2])
  fig.update_yaxes(range = [df['後頭部対称率'].min()-2,102])

  #width = 800*(i+1)
  width = 800*len(df['ヘルメット'].unique())

  fig.update_layout(height=800, width=width, title='短頭率と後頭部対称率の治療前後の変化')

  for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

  st.plotly_chart(fig)

def animate_CI_CVAI(df0, df):
  colors = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8C33', '#33FFF1', '#8C33FF', '#FF5733', '#57FF33', '#5733FF',
    '#FF3357', '#33FFA1', '#FFA133', '#33FF8C', '#FF338C', '#8CFF33', '#A1FF33', '#338CFF', '#A133FF', '#33A1FF'
  ]

  #df0 = df.drop_duplicates('ダミーID', keep='first')

  df1 = df.drop_duplicates('ダミーID', keep='last')

  common_patients = set(df1['ダミーID'].unique()) & (set(df0['ダミーID'].unique()))

  df = pd.concat([df0, df1])
  df = df[df['ダミーID'].isin(common_patients)]

  #複数のヘルメットを使用している患者を除外
  df_helmet = df[df['ヘルメット'] != '経過観察']
  helmet_counts = df_helmet.groupby('ダミーID')['ヘルメット'].nunique()
  common_patients = helmet_counts[helmet_counts > 1].index.tolist()

  df = df[~df['ダミーID'].isin(common_patients)]

  fig = px.scatter(df, x='CI', y='CVAI', color='治療前CVAI重症度', symbol='治療前短頭症', facet_col = 'ヘルメット',
                   hover_data=['ダミーID', '治療期間', '治療前月齢', 'ヘルメット'] + parameters, category_orders=category_orders, animation_frame='治療ステータス', animation_group='ダミーID', color_discrete_sequence=colors)
  i=0
  for i in range(len(df['ヘルメット'].unique())):
    #短頭率の正常範囲
    fig.add_trace(go.Scatter(x=[80, 80], y=[df['CVAI'].min(), 100], mode='lines', line=dict(color='gray', dash = 'dot'), name='CI正常下限'), row=1, col=i+1)
    fig.add_trace(go.Scatter(x=[94, 94], y=[df['CVAI'].min(), 100], mode='lines', line=dict(color='gray', dash = 'dot'), name='CI正常上限'), row=1, col=i+1)

    #CVAIの正常範囲
    fig.add_trace(go.Scatter(x=[df['CI'].min(), df['CI'].max()], y=[5, 5], mode='lines', line=dict(color='gray', dash = 'dot'), name='CVAI正常下限'), row=1, col=i+1)

  fig.update_xaxes(range = [df['CI'].min()-2,df['CI'].max()+2])
  fig.update_yaxes(range = [-2,df['CVAI'].max()])

  #width = 800*(i+1)
  width = 800*len(df['ヘルメット'].unique())

  fig.update_layout(height=800, width=width, title='CIとCVAIの治療前後の変化')

  for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

  st.plotly_chart(fig)

levels = {'短頭率':'治療前短頭症',
          '前頭部対称率':'治療前ASRレベル',
          'CA':'治療前CA重症度',
          '後頭部対称率':'治療前PSRレベル',
          'CVAI':'治療前CVAI重症度',
          'CI':'治療前短頭症'}

borders = {'短頭率':[106, 106],
          '前頭部対称率':[90, 90],
          'CA':[6, 6],
          '後頭部対称率':[90, 90],
          'CVAI':[5, 5],
          'CI':[94, 94]}

colors = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8C33', '#33FFF1', '#8C33FF', '#FF5733', '#57FF33', '#5733FF',
    '#FF3357', '#33FFA1', '#FFA133', '#33FF8C', '#FF338C', '#8CFF33', '#A1FF33', '#338CFF', '#A133FF', '#33A1FF'
  ]

def animate(parameter, df0, df):
  #df0 = df.drop_duplicates('ダミーID', keep='first')

  df1 = df.drop_duplicates('ダミーID', keep='last')

  common_patients = set(df1['ダミーID'].unique()) & (set(df0['ダミーID'].unique()))

  df = pd.concat([df0, df1])
  df = df[df['ダミーID'].isin(common_patients)]

  #複数のヘルメットを使用している患者を除外
  df_helmet = df[df['ヘルメット'] != '経過観察']
  helmet_counts = df_helmet.groupby('ダミーID')['ヘルメット'].nunique()
  common_patients = helmet_counts[helmet_counts > 1].index.tolist()

  df = df[~df['ダミーID'].isin(common_patients)]

  fig = px.scatter(df, x='月齢', y=parameter, color=levels[parameter], symbol = '治療前の月齢', facet_col = 'ヘルメット',
                   hover_data=['ダミーID', '治療期間', '治療前月齢', 'ヘルメット'] + parameters, category_orders=category_orders, animation_frame='治療ステータス', animation_group='ダミーID', color_discrete_sequence=colors)
  i=0
  for i in range(len(df['ヘルメット'].unique())):
    #正常範囲
    fig.add_trace(go.Scatter(x=[df['月齢'].min(), df['月齢'].max()], y=borders[parameter], mode='lines', line=dict(color='gray', dash = 'dot'), name=parameter+'の正常との境界'), row=1, col=i+1)

  fig.update_xaxes(range = [df['月齢'].min()-2,df['月齢'].max()+2])
  fig.update_yaxes(range = [df[parameter].min()-2,df[parameter].max()+2])

  #width = 800*(i+1)
  width = 800*len(df['ヘルメット'].unique())

  fig.update_layout(height=800, width=width, title=parameter+'の治療前後の変化')

  for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

  st.plotly_chart(fig)

def animate_hc(df0, df):
  df_gc = pd.read_csv('成長曲線.csv')
  
  #df0 = df.drop_duplicates('ダミーID', keep='first')

  df1 = df.drop_duplicates('ダミーID', keep='last')

  common_patients = set(df1['ダミーID'].unique()) & (set(df0['ダミーID'].unique()))

  df = pd.concat([df0, df1])
  df = df[df['ダミーID'].isin(common_patients)]

  #複数のヘルメットを使用している患者を除外
  df_helmet = df[df['ヘルメット'] != '経過観察']
  helmet_counts = df_helmet.groupby('ダミーID')['ヘルメット'].nunique()
  common_patients = helmet_counts[helmet_counts > 1].index.tolist()

  df = df[~df['ダミーID'].isin(common_patients)]

  fig = px.scatter(df, x='月齢', y='頭囲', symbol = '治療前の月齢', facet_col = 'ヘルメット',
                   hover_data=['ダミーID', '治療期間', '治療前月齢', 'ヘルメット'] + parameters, category_orders=category_orders, animation_frame='治療ステータス', animation_group='ダミーID', color_discrete_sequence=colors)
  i=0
  for i in range(len(df['ヘルメット'].unique())):
    #正常範囲
    # fig.add_trace(go.Scatter(x=[df['月齢'].min(), df['月齢'].max()], y=borders[parameter], mode='lines', line=dict(color='gray', dash = 'dot'), name=parameter+'の正常との境界'), row=1, col=i+1)

    #成長曲線
    fig_px = px.line(df_gc, x='月齢', y='頭囲', color='sex', line_group='name')
    for trace in fig_px.data:
      fig.add_trace(trace,  row=1, col=i+1)

  fig.update_xaxes(range = [df['月齢'].min()-2,df['月齢'].max()+2])
  fig.update_yaxes(range = [df['頭囲'].min()-2,df['頭囲'].max()+2])

  #width = 800*(i+1)
  width = 800*len(df['ヘルメット'].unique())

  fig.update_layout(height=800, width=width, title='頭囲の治療前後の変化')

  for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

  st.plotly_chart(fig)

def line_plot(parameter, df):
  df_fig = df.copy()
  if '治療前の月齢' not in df_fig.columns:
    df_fig['初診時の月齢'] = df_fig['治療前月齢'].apply(lambda x: np.floor(x) if pd.notnull(x) else np.nan)
    symbol = '初診時の月齢'
  else:
    symbol = '治療前の月齢'

  too_young = df_fig[df_fig['月齢'] < 0]['ダミーID'].unique()
  df_fig = df_fig[~df_fig['ダミーID'].isin(too_young)]

  if parameter == '頭囲':
    fig = px.line(df_fig, x='月齢', y=parameter, line_group='ダミーID')
  else:
    fig = px.line(df_fig, x='月齢', y=parameter, line_group='ダミーID', color=levels[parameter], symbol = symbol, category_orders=category_orders, color_discrete_sequence=colors)

  fig.update_xaxes(range = [df['月齢'].min()-2,df['月齢'].max()+2])
  fig.update_yaxes(range = [df[parameter].min()-2,df[parameter].max()+2])
  fig.update_layout(width=900, title='経過観察前後の' + parameter + 'の変化')

  st.plotly_chart(fig)

# 95%信頼区間を計算する関数
def calc_ci(group):
    mean = group.mean()
    std = group.std()
    n = len(group)
    se = std / np.sqrt(n)

    # 95%信頼区間を計算
    ci_lower, ci_upper = stats.t.interval(0.95, n-1, loc=mean, scale=se)

    return mean, std, se, ci_lower, ci_upper

def make_table(parameter, df, co = False):
  if not co:
    df_temp = df[df['ヘルメット'] != '経過観察']
  else:
    df_temp = df.copy()
  df_temp = df_temp.sort_values('月齢')
  df_temp = df_temp[['ダミーID', '月齢', parameter, '治療前の月齢', levels[parameter], 'ヘルメット']]
  df_before = df_temp.drop_duplicates('ダミーID', keep='first')
  df_before = df_before.rename(columns={parameter:'治療前'+parameter, '月齢':'治療前月齢'})
  df_before = df_before[['ダミーID', '治療前'+parameter, '治療前月齢']]

  df_after = df_temp.drop_duplicates('ダミーID', keep='last')
  df_after = df_after.rename(columns={parameter:'治療後'+parameter, '月齢':'治療後月齢'})

  df_before_after = pd.merge(df_before, df_after, on='ダミーID', how='left')

  df_before_after['変化量'] = df_before_after['治療後'+parameter] - df_before_after['治療前'+parameter]
  df_before_after['治療期間'] = df_before_after['治療後月齢'] - df_before_after['治療前月齢']

  df_before_after[levels[parameter]] = pd.Categorical(df_before_after[levels[parameter]],
                                    categories=category_orders[levels[parameter]],
                                    ordered=True)

  # 指定した順序でgroupbyし、変化量に対して各種統計量を計算
  result = df_before_after.groupby(['治療前の月齢', levels[parameter]], observed=False).agg(
      mean=('変化量', 'mean'),
      std=('変化量', 'std'),
      count=('変化量', 'count'),
      min=('変化量', 'min'),
      max=('変化量', 'max'),
      mean_d=('治療期間', 'mean'),
      std_d=('治療期間', 'std'),
      min_d=('治療期間', 'min'),
      max_d=('治療期間', 'max')
  )

  # 標準誤差と95%信頼区間を計算してカラムに追加
  result['se'] = result['std'] / np.sqrt(result['count'])
  result['95% CI lower'], result['95% CI upper'] = stats.t.interval(
      0.95, result['count']-1, loc=result['mean'], scale=result['se']
  )
  result['se_d'] = result['std_d'] / np.sqrt(result['count'])
  result['95% CI lower_d'], result['95% CI upper_d'] = stats.t.interval(
      0.95, result['count']-1, loc=result['mean_d'], scale=result['se_d']
  )

  # 小数点以下2桁に丸める
  result = result.round(2)

  # 結果表示
  #import ace_tools as tools; tools.display_dataframe_to_user(name="信頼区間を含む統計結果", dataframe=result)
  result = result.rename(columns={'mean':'平均', 'std':'標準偏差', 'count':'人数', 'se':'標準誤差', 'min':'最小', 'max':'最大',
                                  'mean_d':'平均治療期間', 'std_d':'標準偏差 ', 'se_d':'標準誤差 ', 'min_d':'最小 ', 'max_d':'最大 '})
  result = result.replace(np.nan, '-')
  result['95% 信頼区間'] = result['95% CI lower'].astype(str) + ' ～ ' + result['95% CI upper'].astype(str)
  result['95% 信頼区間 '] = result['95% CI lower_d'].astype(str) + ' ～ ' + result['95% CI upper_d'].astype(str)
  result = result[['平均', '95% 信頼区間', '標準偏差', '最小', '最大', '人数', '平均治療期間', '95% 信頼区間 ', '標準偏差 ', '最小 ', '最大 ']]
  result = result.reset_index()
  result['治療前の月齢'] = result['治療前の月齢'].astype(int)

  if co:
    result = result.rename(columns={levels[parameter]:'初診時'+parameter, '治療前の月齢':'初診時の月齢', '平均治療期間': '平均受診間隔'})

  return (result)

def make_confusion_matrix(df, parameter):
  parameter_category_names = {'短頭率': '短頭症', '前頭部対称率':'ASRレベル', 'CA':'CA重症度', '後頭部対称率':'PSRレベル', 'CVAI':'CVAI重症度', 'CI':'短頭症'}
  parameter_category_name = parameter_category_names[parameter]

  order = category_orders['治療前'+parameter_category_name]

  for_pivot_df = df.drop_duplicates('ダミーID')

  pivot_table = for_pivot_df.pivot_table(index="治療前" + parameter_category_name, columns="最終" + parameter_category_name, aggfunc="size", fill_value=0)

  # 各行の合計を計算
  pivot_table["Total"] = pivot_table.sum(axis=1)

  # 割合（行方向の合計で割る）
  pivot_table_percentage = 2 * pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

  # % 付きの文字列に変換
  pivot_table_percentage = pivot_table_percentage.round(1).astype(str) + "%"

  # 人数と割合を結合
  pivot_table_combined = pivot_table.astype(str) + " (" + pivot_table_percentage + ")"

  pivot_table_combined["Total"] = pivot_table["Total"].astype(str)

  df0 = df.drop_duplicates('ダミーID', keep='first').sort_values('ダミーID').reset_index(drop=True)
  df1 = df.drop_duplicates('ダミーID', keep='last').sort_values('ダミーID').reset_index(drop=True)
  df_delta = df0.copy()
  df_delta['変化量'] = df1[parameter] - df0[parameter]
  
  pivot_table_combined['変化量'] = df_delta.groupby("治療前" + parameter_category_name)['変化量'].mean().round(2).astype(str) + " ± " + df_delta.groupby("治療前" + parameter_category_name)['変化量'].std().round(2).astype(str)
  
  pivot_table_combined = pivot_table_combined.reindex(index=order, columns=order + ["Total", "変化量"])

  pivot_table_combined = pivot_table_combined.fillna('0 (0.0%)')

  return(pivot_table_combined)

def animate_CI_CVAI_over_age(df_co):
  # 元データを整理
  df = df_co.copy()
  df = df.sort_values(['ダミーID', '月齢'])
  df = df.dropna(subset=['月齢', 'CI', 'CVAI'])

  interp_list = []

  for dummy_id, g in df.groupby('ダミーID'):
    # 時系列が 2 点未満の患者は補間できないのでスキップ
    if g['月齢'].nunique() < 2:
      continue

    ages = g['月齢'].values
    ci = g['CI'].values
    cvai = g['CVAI'].values

    # この患者の月齢範囲で 0.1 か月刻みの軸をつくる
    age_new = np.arange(ages.min(), ages.max() + 1e-6, 0.1)

    # CI / CVAI を線形補間
    ci_new = np.interp(age_new, ages, ci)
    cvai_new = np.interp(age_new, ages, cvai)

    # 色・シンボル用の情報は最初の行から引き継ぐ
    base = g.iloc[0]

    interp_list.append(pd.DataFrame({
        'ダミーID': dummy_id,
        '月齢_interp': age_new,
        'CI': ci_new,
        'CVAI': cvai_new,
        '治療前CVAI重症度': base['治療前CVAI重症度'],
        '治療前短頭症': base['治療前短頭症'],
        '治療前の月齢': base['治療前の月齢'],
    }))

  if len(interp_list) == 0:
    st.write('補間できる患者がいません')
    return

  df_anim = pd.concat(interp_list, ignore_index=True)

  # フレーム用に小数第1位に丸めた月齢を使う
  df_anim['月齢_frame'] = df_anim['月齢_interp'].round(1)

  fig = px.scatter(
      df_anim,
      x='CVAI',
      y='CI',
      animation_frame='月齢_frame',     # 月齢 0.1 刻みのフレーム
      animation_group='ダミーID',       # 患者ごとに軌跡をつなぐ
      color='治療前CVAI重症度',        # 色分け（お好みで変更可）
      symbol='治療前短頭症',           # マーカー形状（お好みで変更可）
      hover_data=['ダミーID', '月齢_interp', '治療前の月齢'],
      category_orders=category_orders,
      color_discrete_sequence=colors
  )

  # 正常範囲のガイドライン（不要なら削除）
  fig.add_hline(y=80, line_dash='dot', line_color='gray', name='CI=80')
  fig.add_hline(y=94, line_dash='dot', line_color='gray', name='CI=94')
  fig.add_vline(x=5,  line_dash='dot', line_color='gray', name='CVAI=5')

  # すべてのフレームで軸スケールを固定
  fig.update_xaxes(
      title='CVAI',
      # range=[df_anim['CVAI'].min() - 1, df_anim['CVAI'].max() + 1]
    range=[0, df_anim['CVAI'].max() + 1]
  )
  fig.update_yaxes(
      title='CI',
      # range=[df_anim['CI'].min() - 1, df_anim['CI'].max() + 1]
    range=[70, 110]
  )

  fig.update_layout(
      width=900,
      height=800,
      title='CI–CVAI 経過観察（月齢 0.1 か月刻み補間）',
      plot_bgcolor='white'
  )

  st.plotly_chart(fig)

##関数パート終了

st.markdown('<div style="text-align: left; color:black; font-size:36px; font-weight: bold;">位置的頭蓋変形の診療に関するデータビジュアライゼーション</div>', unsafe_allow_html=True)

from datetime import datetime, timedelta

# 昨日の日付を取得
yesterday = datetime.now() - timedelta(days=1)

# YYYY年MM月DD日形式でフォーマット
formatted_date = yesterday.strftime("%Y年%m月%d日")

st.markdown(f'<div style="text-align: left; color:black; font-size:18px;">以下のグラフは2021年03月04日から{formatted_date}までのデータにもとづいています</div>', unsafe_allow_html=True)
#st.write('以下のグラフは2021年03月04日から' + formatted_date + 'までのデータにもとづいています')

st.plotly_chart(fig)

st.write('')
st.write('')
st.markdown("---")
st.markdown('<div style="text-align: left; color:black; font-size:24px; font-weight: bold;">受診患者の重症度の分布および矯正治療を受けた割合</div>', unsafe_allow_html=True)

parameters = ['短頭率', '前頭部対称率', '後頭部対称率', 'CA', 'CVAI', 'CI']

for parameter in parameters:
  hist(parameter)
  st.markdown("---")

show_helmet_proportion()
st.markdown("---")

show_age_proportion(df_tx_pre_post)
st.markdown("---")

st.markdown('<div style="text-align: left; color:black; font-size:24px; font-weight: bold;">月齢・重症度別の治療前後の変化</div>', unsafe_allow_html=True)
st.write('以下のグラフと表は全てのヘルメットを合わせたものです')

table_members = df_tx_pre_post[df_tx_pre_post['治療期間'] > 1]['ダミーID'].unique()
df_table = df_tx_pre_post[df_tx_pre_post['ダミーID'].isin(table_members)]

for parameter in parameters:
  st.write('')
  st.write('')
  st.write(parameter+'の治療前後の変化（1か月以上の治療）')
  graham(df_table, parameter)

  result = make_confusion_matrix(df_table, parameter)
  st.dataframe(result, width=800)
  
  result = make_table(parameter, df_table)
  #st.table(result)
  st.dataframe(result, width=800)
  st.markdown("---")

st.write('')
st.write('')
st.write('頭囲の治療前後の変化（1か月以上の治療）')
graham_hc(df_table)

#result = make_table('頭囲', df_table)
#st.table(result)
#st.dataframe(result, width=800)
st.markdown("---")

#df_vis = takamatsu(df_tx)
#st.dataframe(df_vis)
#st.table(df_vis)

with st.form(key='filter_form'):
  st.write('患者を絞ってグラフを作成します')

  # スライダーで範囲を指定
  min_age, max_age = st.slider(
      '月齢の範囲を選択してください',
      min_value = max([int(df_tx_pre_post['治療前月齢'].min()),1]),
      max_value = int(df_tx_pre_post['治療前月齢'].max()),
      value=( max([int(df_tx_pre_post['治療前月齢'].min()),1]), int(df_tx_pre_post['治療前月齢'].max()))
  )

  min_value, max_value = st.slider(
      '治療期間（治療前スキャン〜治療後スキャンの間隔）の範囲を選択してください',
      min_value = max([int(df_tx_pre_post['治療期間'].min()),1]),
      #max_value = int(df_tx_pre_post['治療期間'].max()),
      max_value = 12,
      #value=(max([int(df_tx_pre_post['治療期間'].min()),1]), int(df_tx_pre_post['治療期間'].max()))
      value=(max([int(df_tx_pre_post['治療期間'].min()),1]), 12)
  )

  st.write('ヘルメットを選択してください（複数選択可）')

  # チェックボックスを作成
  filter_pass0 = st.checkbox('アイメット')
  filter_pass1 = st.checkbox('クルム')
  filter_pass2 = st.checkbox('クルムフィット')
  filter_pass3 = st.checkbox('経過観察')

  submit_button = st.form_submit_button(label='実行')

# 「実行」ボタンを作成
#if st.button('実行'):
if submit_button:
  if not filter_pass0 and not filter_pass1 and not filter_pass2 and not filter_pass3:
    st.write('一つ以上のチェックボックスを選択してください')
  else:
    st.write('選択された治療期間（治療前スキャン〜治療後スキャンの間隔）：', str(min_value), "〜", str(max_value), "か月")
    
    filtered_df = df_tx_pre_post[df_tx_pre_post['治療ステータス'] == '治療後']
        # スライダーで選択された範囲でデータをフィルタリング
    filtered_df_first = df_first[(df_first['月齢'] >= min_age) & (df_first['月齢'] <= max_age)]
    filtered_df = filtered_df[(filtered_df['治療前月齢'] >= min_age) & (filtered_df['治療前月齢'] <= max_age)]
    filtered_df_co = df_co[(df_co['治療前月齢'] >= min_age) & (df_co['治療前月齢'] <= max_age)]
    filtered_df_tx_pre_post = df_tx_pre_post[(df_tx_pre_post['治療前月齢'] >= min_age) & (df_tx_pre_post['治療前月齢'] <= max_age)]

    filtered_df = filtered_df[(filtered_df['治療期間'] >= min_value) & (filtered_df['治療期間'] <= max_value)]
    filtered_df_co = filtered_df_co[(filtered_df_co['治療期間'] >= min_value) & (filtered_df_co['治療期間'] <= max_value)]

    filtered_table_members = filtered_df_tx_pre_post[(filtered_df_tx_pre_post['治療期間'] >= min_value) & (filtered_df_tx_pre_post['治療期間'] <= max_value)]['ダミーID'].unique()
    filtered_df_tx_pre_post = filtered_df_tx_pre_post[filtered_df_tx_pre_post['治療期間'] <= max_value]
    filtered_df_tx_pre_post = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ダミーID'].isin(filtered_table_members)]

    filtered_df = filtered_df[(filtered_df['治療期間'] >= min_value) & (filtered_df['治療期間'] <= max_value)]

    filtered_df0 = df_tx_pre_post[df_tx_pre_post['治療ステータス'] == '治療前']

    # チェックボックスの状態に応じてデータをフィルタリング
    if not filter_pass0:
        filtered_df = filtered_df[filtered_df['ヘルメット'] != 'アイメット']
        filtered_df0 = filtered_df0[filtered_df0['ヘルメット'] != 'アイメット']
        filtered_df_tx_pre_post = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] != 'アイメット']
    if not filter_pass1:
        filtered_df = filtered_df[filtered_df['ヘルメット'] != 'クルム']
        filtered_df0 = filtered_df0[filtered_df0['ヘルメット'] != 'クルム']
        filtered_df_tx_pre_post = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] != 'クルム']
    if not filter_pass2:
        filtered_df = filtered_df[filtered_df['ヘルメット'] != 'クルムフィット']
        filtered_df0 = filtered_df0[filtered_df0['ヘルメット'] != 'クルムフィット']
        filtered_df_tx_pre_post = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] != 'クルムフィット']
    if not filter_pass3:
        filtered_df = filtered_df[filtered_df['ヘルメット'] != '経過観察']
        filtered_df0 = filtered_df0[filtered_df0['ヘルメット'] != '経過観察']
        filtered_df_tx_pre_post = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] != '経過観察']


    filtered_treated_patients = filtered_df['ダミーID'].unique()
    filtered_df = filtered_df[filtered_df['ダミーID'].isin(filtered_treated_patients)]
    filtered_df0 = filtered_df0[filtered_df0['ダミーID'].isin(filtered_treated_patients)]


    st.write('▶を押すと治療前後の変化が見られます。')
    animate_BI_PSR(filtered_df0, filtered_df)
    st.markdown("---")

    animate_CI_CVAI(filtered_df0, filtered_df)
    st.markdown("---")

    animate_hc(filtered_df0, filtered_df)
    st.markdown("---")
    
    for parameter in parameters:
      animate(parameter, filtered_df0, filtered_df)
      st.markdown("---")

    if (min_age != 1) | (max_age != 13):
      st.markdown("---")
      st.write('対象を制限した場合のヒストグラムを表示します')
      for parameter in parameters:
        hist(parameter, filtered_df_first)
        st.markdown("---")

    filtered_treated_patients = filtered_df_tx_pre_post[filtered_df_tx_pre_post['治療ステータス'] == '治療後']['ダミーID'].unique()
    filtered_df_tx_pre_post = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ダミーID'].isin(filtered_treated_patients)]
    
    if filter_pass0 | filter_pass1 | filter_pass2:
      for parameter in parameters:
        count = len(filtered_df_tx_pre_post['ダミーID'].unique())
        st.write('')
        st.write('')
        st.write(parameter+'の治療前後の変化　', str(count), '人')
        graham(filtered_df_tx_pre_post, parameter, x_limit=max_value)
        result = make_confusion_matrix(filtered_df_tx_pre_post, parameter)
        st.dataframe(result, width=800)
        result = make_table(parameter, filtered_df_tx_pre_post)
        st.dataframe(result, width=800)
        st.markdown("---")

        if filter_pass0:
          filtered_df_helmet = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] == 'アイメット']
          count = len(filtered_df_helmet['ダミーID'].unique())
          st.write('')
          st.write('')
          st.write(parameter+'の治療前後の変化(アイメット)　', str(count), '人')
          graham(filtered_df_helmet, parameter, x_limit=max_value)
          result = make_confusion_matrix(filtered_df_helmet, parameter)
          st.dataframe(result, width=800)
          result = make_table(parameter, filtered_df_helmet)
          st.dataframe(result, width=800)
          st.markdown("---")

        if filter_pass1:
          filtered_df_helmet = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] == 'クルム']
          count = len(filtered_df_helmet['ダミーID'].unique())
          st.write('')
          st.write('')
          st.write(parameter+'の治療前後の変化(クルム)　', str(count), '人')
          graham(filtered_df_helmet, parameter, x_limit=max_value)
          result = make_confusion_matrix(filtered_df_helmet, parameter)
          st.dataframe(result, width=800)
          result = make_table(parameter, filtered_df_helmet)
          st.dataframe(result, width=800)
          st.markdown("---")

        if filter_pass2:
          filtered_df_helmet = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] == 'クルムフィット']
          count = len(filtered_df_helmet['ダミーID'].unique())
          st.write('')
          st.write('')
          st.write(parameter+'の治療前後の変化(クルムフィット)　', str(count), '人')
          graham(filtered_df_helmet, parameter, x_limit=max_value)
          result = make_confusion_matrix(filtered_df_helmet, parameter)
          st.dataframe(result, width=800)
          result = make_table(parameter, filtered_df_helmet)
          st.dataframe(result, width=800)
          st.markdown("---")

      count = len(filtered_df_tx_pre_post['ダミーID'].unique())
      st.write('')
      st.write('')
      st.write('頭囲の治療前後の変化　', str(count), '人')
      graham_hc(filtered_df_tx_pre_post, x_limit=max_value)
      #result = make_table(parameter, filtered_df_tx_pre_post)
      #st.dataframe(result, width=800)
      st.markdown("---")

      if filter_pass0:
        filtered_df_helmet = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] == 'アイメット']
        count = len(filtered_df_helmet['ダミーID'].unique())
        st.write('')
        st.write('')
        st.write('頭囲の治療前後の変化(アイメット)　', str(count), '人')
        graham_hc(filtered_df_helmet, x_limit=max_value)
        #result = make_table('頭囲', filtered_df_helmet)
        #st.dataframe(result, width=800)
        st.markdown("---")

      if filter_pass1:
        filtered_df_helmet = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] == 'クルム']
        count = len(filtered_df_helmet['ダミーID'].unique())
        st.write('')
        st.write('')
        st.write('頭囲の治療前後の変化(クルム)　', str(count), '人')
        graham_hc(filtered_df_helmet, x_limit=max_value)
        #result = make_table('頭囲', filtered_df_helmet)
        #st.dataframe(result, width=800)
        st.markdown("---")

      if filter_pass2:
        filtered_df_helmet = filtered_df_tx_pre_post[filtered_df_tx_pre_post['ヘルメット'] == 'クルムフィット']
        count = len(filtered_df_helmet['ダミーID'].unique())
        st.write('')
        st.write('')
        st.write('頭囲の治療前後の変化(クルムフィット)　', str(count), '人')
        graham_hc(filtered_df_helmet, x_limit=max_value)
        #result = make_table('頭囲', filtered_df_helmet)
        #st.dataframe(result, width=800)
        st.markdown("---")
    
    if filter_pass3:
      st.write('経過観察した場合のグラフを表示します')
      count = len(filtered_df_co['ダミーID'].unique())
      st.write(str(count), '人')
      st.markdown('<div style="text-align: left; color:black; font-size:24px; font-weight: bold;">経過観察群の CI–CVAI の推移</div>', unsafe_allow_html=True)
      animate_CI_CVAI_over_age(filtered_df_co)
      #st.dataframe(filtered_df_co, width=800)
      for parameter in parameters:
        st.write('')
        st.write('')
        line_plot(parameter, filtered_df_co)

        graham(filtered_df_co, parameter)
        
        result = make_confusion_matrix(filtered_df_co, parameter)
        st.dataframe(result, width=800)
        
        result = make_table(parameter, filtered_df_co, co = True)
        #st.table(result)
        st.dataframe(result, width=800)
        st.markdown("---")

      st.write('')
      st.write('')
      line_plot('頭囲', filtered_df_co)

      graham_hc(filtered_df_co)
      #result = make_table('頭囲', filtered_df_co, co = True)
      #st.table(result)
      #st.dataframe(result, width=800)
      st.markdown("---")

    #df_vis = takamatsu(filtered_df_tx_pre_post)
    #st.dataframe(df_vis)
    #st.table(df_vis)
else:
    st.write('実行ボタンを押すとグラフが作成されます')
