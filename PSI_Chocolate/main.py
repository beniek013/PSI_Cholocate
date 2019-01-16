import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#przygotowanie danych
data = pd.read_csv("flavors_of_cacao.csv")
data.columns =data.columns.str.replace('\n', ' ').str.replace('\xa0', '')
#print(data.info())
data.fillna(0,inplace=True)
#print(data.isnull().sum())
data['CocoaPercent']=data['CocoaPercent'].apply(lambda x: x[:-1]).astype('float')
#data.info()
print(data.corr())
sns.heatmap(data.corr())
plt.savefig('corelation.png')