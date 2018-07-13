# hello-world
新的世界啊
# -*- coding: utf-8 -*-
#using python 3
# import pandas as pd
# df = pd.read_csv('E:/preddata.csv',index = None)
# name = [['动点点击','数字点击','打地鼠','打地鼠-II']]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import  Embedding,Dropout,Dense,Reshape,Merge
from sqlalchemy import create_engine
import pymysql
import random




#数据预处理
k=128
pymysql.install_as_MySQLdb()
# 创建mysql连接引擎
# engine = create_engine('mysql+mysqldb://username:password@host:port/dbname?charset=utf8')
#查询数据并转为pandas.DataFrame，指定DataFrame的index为数据库中的id字段
engine = create_engine(
    'mysql+mysqldb://root:root@192.168.10.14:3306/com66nao_cloud?charset=utf8')
# df = pd.read_sql('SELECT * FROM game1', engine, index_col='id')
#将修改后的数据追加至原表,index=False代表不插入索引，因为数据库中id字段为自增字段
df4 = pd.read_sql('select user_id as user_truename,train_score as score ,cogn_task.name as name ,cloud_cat.name as firstbrain from user_train_history join cogn_task on cogn_task.id=user_train_history.game_id join cloud_cat on cloud_cat.id=cogn_task.label', engine)



df1 = df4[['user_truename', 'name', 'score']]  # 只选取有实际作用的列
grouped = df1['score'].groupby([df1['user_truename'], df1['name']])
# print(grouped)
df2 = grouped.median()
# print(df2)
df3 = df2.reset_index()

print('ok')
old_set=np.unique(df3['user_truename'])
old_list=list(old_set)
new_id=np.arange(len(old_list))
df3['new_id']=df3['user_truename'].replace(old_list,new_id)
df3['new_id']=df3['new_id']+1
df3
print('OK')

old_gameset = np.unique(df3['name'])
old_gamelist = list(old_gameset)
fid = np.arange(len(old_gamelist))
df3['fid'] = df3['name'].replace(old_gamelist,fid)
df3['fid'] = df3['fid']+1

#找到敏捷性四个游戏在predata中的位置，将其提取出来
name = ['动点点击','数字点击','打地鼠','打地鼠-II']
namelist = [i for i in range(len(old_gameset)) if old_gameset[i] in name]
data = pd.read_csv('E:/preddata.csv')
namestr = [str(e) for e in namelist]
print(data[namestr])
minjiexing_data = np.array(data[namestr])
print('ok')





from keras.models import model_from_json
json_file = open('minjiexingmodel711.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("minjiexingmodel711.h5")

preditions  = model.predict(minjiexing_data)
print('ok')
