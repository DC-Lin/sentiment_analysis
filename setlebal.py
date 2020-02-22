from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import re
import pandas as pd
from torch import nn,functional as F
import torch

class Setlabel:
    def clearData(self):
        datas=pd.read_csv('datas.csv')
        filter_pattern = re.compile('[^\u4e00-\u9fa5,。.，?!;；：:a-zA-Z0-9]+$')
        filter_pattern1=re.compile('<.*>')
        for i,value in enumerate(datas['评论']):
            datas['评论'][i]=filter_pattern1.sub('',datas['评论'][i])
            datas['评论'][i] = filter_pattern.sub('', datas['评论'][i])
            if len(datas['评论'][i])<2:
                datas=datas.drop(i)
        datas=datas.drop_duplicates(['评论'],keep='first',)
        datas.to_csv('Cdatas1.csv',index=False)

    def kms(self,datas):#kms聚类
        Scores=[]
        for k in range(5,11):
            KMS=KMeans(n_clusters=k)
            KMS.fit(datas)
            Scores.append(silhouette_score(datas,KMS.labels_,metric='euclidean'))
        plt.xlabel('k')
        plt.ylabel('轮廓系数')
        plt.plot(range(2,9),Scores,'o-')
        plt.imsave('picture/BestK.jpg')
    def dbs(self):#dbscan聚类
        datas=pd.read_csv('Cdatas1.csv')
        embed=nn.Embedding(10000,100)
        datas=embed(torch.Tensor([datas['评论']]))
        print(datas)
        # db=DBSCAN(eps=10,min_samples=30)
        # db.fit(datas)
        # labels=db.labels_
        # datas['labels']=labels
        # datas.groupby('labels').count()

    def plotFeature(data, clusters, clusterNum):
        nPoints = data.shape[1] #label
        matClusters = np.mat(clusters).transpose()
        fig = plt.figure()
        scatterColors = ['turquoise','violet','quartz','cadetblue','coral','darkmagenta','black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown'][:clusterNum]
        ax = fig.add_subplot(111)
        for i in range(clusterNum + 1):
            colorSytle = scatterColors[i % len(scatterColors)]
            subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
            ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50)
        plt.imsave('picture/分类效果图.jpg')
    def worldembedding(self):
        torch
    def setlabe(self):
        datas=pd.read_csv('trainSet.csv',sep='\t')
        for v,i in enumerate(datas.iterrows()):
            if v%10==0:
                print(datas.columns.values)
            print(datas['text_a'][v])
            while True:
                try:
                    datas.iloc[v, 2:]=list(input())
                except Exception as e:
                    print("输入有误")
                    continue
                else:
                    break
        datas.to_csv('trainSet.csv',sep='\t',index=False)
    def ran(self):
        datas=pd.read_csv('trainSet.csv',sep='\t')
        import random
        datas['难过']=np.random.randint(0,2,datas.shape[0])
        datas['开心']=np.random.randint(0,2,datas.shape[0])
        datas['赞赏']=np.random.randint(0,2,datas.shape[0])
        datas['批评']=np.random.randint(0,2,datas.shape[0])
        datas['怀疑']=np.random.randint(0,2,datas.shape[0])
        datas['喜欢']=np.random.randint(0,2,datas.shape[0])
        datas['惊叹']=np.random.randint(0,2,datas.shape[0])
        datas['鼓励']=np.random.randint(0,2,datas.shape[0])
        print(datas['text_a'])
        datas.to_csv('trainSet.csv', sep='\t', index=False)

#
# dataset=Setlabel()
# # dataset.clearData()
# dataset.ran()
def CD2():
    datas=pd.read_csv('datase.csv')
    datas.rename(columns={'评论':'text_a'},inplace=True)
    datas.to_csv('Cdatas.csv',index=False)
def splitSet():
    datas = pd.read_csv('enddatas.csv',sep='\t')

    from sklearn.utils import shuffle
    datas = shuffle(datas)
    trainSet=datas.iloc[:-1100,:]
    testSet=datas.iloc[-1100:-550,:]
    devSet=datas.iloc[-550:-100,:]
    preSet=datas.iloc[:-100,:0]
    trainSet.to_csv('trainSet.csv',index=False,sep='\t')
    devSet.to_csv('devSet.csv',index=False,sep='\t')
    testSet.to_csv('testSet.csv',index=False,sep='\t')
    preSet.to_csv('preSet.csv',index=False,sep='\t')
# splitSet()

