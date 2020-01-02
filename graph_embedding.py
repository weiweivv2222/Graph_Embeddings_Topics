# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:52:07 2019

@author: xg16060
"""
#%%
import networkx as nx
from gensim.models import Word2Vec, keyedvectors
from node2vec import Node2Vec
import pandas as pd
import numpy as np
import math
#import xlrd
#%%
#loading data 
dataFolder='C:\\dev\\data\\interestProfile\\'
clicks = pd.read_csv(dataFolder+'Clicks.csv', encoding= 'ISO-8859-1')
customer = pd.read_csv(dataFolder+'Customer_data.csv')
sliver_customer = pd.read_excel(dataFolder+'PreprocessedData2.xlsx')
province=pd.read_csv(dataFolder+'provincie.csv',header=0,sep=';',encoding= 'ISO-8859-1')
#%%
#select useful cols from preprocessdata
sliver_customer = sliver_customer[['KLTID', 'GIFSF', 'MD', 'OB', 'DZ', 'PK', 'CD_LAND']]

#combine the selected cols with the customer data (kltid and cd_plaats) via klant id
df = pd.merge(sliver_customer, customer[['KLTID', 'CD_PLAATS']], on=['KLTID'], how='left')

#only select customers who live in the netherlands
df = df[df.CD_LAND == 6030]
df=df.dropna()
df[['CD_PLAATS']]=df[['CD_PLAATS']].astype(int)
tags = ['GIFSF', 'MD', 'OB', 'DZ', 'PK']
#remove 0 rows
df=df.loc[df.iloc[:,1:6].sum(axis=1) !=0]

##load province table from the city hall
province=province[['pc2','PROVINCIE']]
province=province.rename(columns={'pc2':'CD_PLAATS'})

#left join province table with df
df=pd.merge(df,province,on=['CD_PLAATS'],how='left')

#remove missing province
df=df.dropna(subset=['PROVINCIE'])
#%%
province_freq= {}
for i in range(len(df)):
    if i in df.index:
        if df['PROVINCIE'][i] not in province_freq:
            province_freq[df['PROVINCIE'][i]] = [0] * len(tags)
        for j in range(len(tags)):
            province_freq[df['PROVINCIE'][i]][j] += df[tags[j]][i]
#%%
prob_province = {}
for province, value in province_freq.items():
    X = np.array(value)
    sum1 = np.sum(X)
    prob = np.round(X/sum1, 3)
    prob_province[province] = prob
 #%% making the neighbor connects
province_neighbors = {}
for province in province_freq.keys():
    #print(province)
    if province=='Noord Brabant':
        province_neighbors[province] = ['Limburg', 'Zuid Holland', 'Zeeland', 'Gelderland']
    if province=='Overijssel':
         province_neighbors[province] = ['Drenthe', 'Frysl\x83n','Gelderland', 'Flevoland']
    if province=='Gelderland':
         province_neighbors[province] = ['Noord Brabant', 'Overijssel','Utrecht', 'Limburg', 'Drenthe', 'Flevoland', 'Zuid Holland']
    if province=='Utrecht':
         province_neighbors[province] = ['Gelderland','Noord Holland', 'Zuid Holland', 'Flevoland']
    if province=='Zuid Holland':
         province_neighbors[province] = ['Noord Holland', 'Zeeland','Gelderland', 'Noord Brabant', 'Utrecht']
    if province=='Noord Holland':
         province_neighbors[province] = ['Zuid Holland', 'Frysl\x83n', 'Utrecht', 'Flevoland']
    if province=='Groningen':
         province_neighbors[province] = ['Drenthe', 'Frysl\x83n']
    if province== 'Limburg':
         province_neighbors[province] = ['Noord Brabant','Gelderland']
    if province=='Drenthe':
         province_neighbors[province] = ['Frysl\x83n','Groningen', 'Overijssel']
    if province=='Flevoland':
         province_neighbors[province] = ['Overijssel', 'Frysl\x83n', 'Noord Holland', 'Utrecht', 'Gelderland']
    if province=='Zeeland':
         province_neighbors[province] = ['Zuid Holland', 'Noord Brabant']
    if province== 'Frysl\x83n':
         province_neighbors[province] = ['Groningen', 'Overijssel', 'Drenthe', 'Noord Holland', 'Flevoland']
#%% Only the Location Graph 
GLocation=nx.Graph()#    
#GLocation.add_edges_from(province_freq) 
for i in province_freq.keys():
    for k in province_neighbors[i]:
        if not GLocation.has_edge(i,k):
            GLocation.add_edge(i,k)
nx.draw_networkx(GLocation)       
    
#%%
#generate a multigraph includes province nodes and topic nodes. 
#Do not consider the difference between the same location with the topic. 
GLocation_topic_Location=nx.Graph()# only has one edges between two nodes
GLocation_topic_Location.add_nodes_from(tags)
#add edges between topics and province with consideration the clients
for i in province_freq.keys():
    for j in tags:
        GLocation_topic_Location.add_edge(i,j)
    # add the neigborhood edges 
    for k in province_neighbors[i]:
        if not GLocation_topic_Location.has_edge(i,k):
            GLocation_topic_Location.add_edge(i,k)
#print(GLocation_topic_Location.edges)
nx.draw_networkx(GLocation_topic_Location)


#circle output
#pos = nx.spring_layout(GLocation_topic_Location, iterations=50, k=500/math.sqrt(GLocation_topic_Location.order()))
#nx.draw(GLocation_topic_Location, pos, with_labels = True)

#%%#generate a multigraph includes province nodes and topic nodes. 
#each edge between province and topic is unique

GLoc_topic_Loc_cust=nx.MultiGraph()
#province

GLoc_topic_Loc_cust.add_nodes_from(tags)

#add edges between topics and province with consideration the clients
for i in province_freq.keys():
    for j,item in enumerate(province_freq[i]):
        for k in range(province_freq[i][j]):
            GLoc_topic_Loc_cust.add_edge(i, tags[j])
            for m in province_neighbors[i]:
                if not GLoc_topic_Loc_cust.has_edge(i,m):
                    GLoc_topic_Loc_cust.add_edge(i,m)
                    
print(GLoc_topic_Loc_cust.edges)
pos = nx.spring_layout(GLoc_topic_Loc_cust, iterations=50, k=500/math.sqrt(GLoc_topic_Loc_cust.order()))

nx.draw(GLoc_topic_Loc_cust, pos, with_labels = True)
#nx.draw(GLoc_topic_Loc_cust, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

#print the edges between two specific nodes
#print(GLoc_topic_Loc_cust['PK'] ['Frysl\x83n'])

#check the edge between province 
#print(GLoc_topic_Loc_cust.has_edge('Limburg', 'Noord Brabant'))
#%%

Gprovince=nx.MultiGraph()
#province
pronames=province_freq.keys()
Gprovince.add_nodes_from(pronames)

#add edges between topics and province with consideration the clients
for i in province_freq.keys():
    for j,item in enumerate(province_freq[i]):
        for k in range(province_freq[i][j]):
            Gprovince.add_edge(i, tags[j])
            
Gprovince = Gprovince.to_undirected()
#print(Gprovince.edges)
nx.draw_networkx(Gprovince)
#%%
##%% TO DO: find the weights between the location nodes.
##generate the empty KG
#GLocationwithweight=nx.Graph()# 
#GLocationwithweight.add_nodes_from(tags)
##add edges between topics and province with consideration the clients
#for i in province_freq.keys():
#    for idx,j in enumerate(tags):
#        GLocationwithweight.add_edge(i,j)
#        GLocationwithweight[i][j]['weight'] = prob_province[i][idx]
#    for k in province_neighbors[i]:
#        if not GLocationwithweight.has_edge(i,k):
#            GLocationwithweight.add_edge(i,k)
## add the neigborhood edges 
#
##print(GLocationwithweight.edges)
#print(GLocationwithweight['GIFSF']['Noord Brabant']['weight'])
#print(GLocationwithweight.has_edge('Limburg', 'Noord Brabant'))


#%%
def dif_graph_embeddings(graph):
    model=Node2Vec(graph, dimensions=64, walk_length=3, num_walks=60, p = 1, q = 1).fit(window=10, min_count=1)
    embeddings_dict={}
    for province in province_freq.keys():
        embeddings_dict[province]=model.wv.get_vector(province)
    # convert dict to dataframe
    embeddings_df=pd.DataFrame(embeddings_dict)
    return embeddings_df
    

#%%
    
node2vec= Node2Vec(Gprovince, dimensions=64, walk_length=3, num_walks=60, p=1,q=1)
# Learn embeddings 
model = node2vec.fit(window=10, min_count=1,batch_words=4)

model.wv.get_vector("Noord Holland")

# Save embeddings for later use
model.save("word2vec.model")
model.wv.save_word2vec_format('model.bin', binary=True)
#Word2Vec.load("word2vec.model")
model.most_similar("Noord Holland")
model.most_similar("OB")
model["Noord Holland"]
#number of nodes and edges
n = Gprovince.number_of_nodes()
m = Gprovince.number_of_edges()
print("Number of nodes :", str(n))
print("Number of edges :", str(m))
print("Number of connected components :" ,str(nx.number_connected_components(Gprovince)))
#%%
#export the embedding of all the location nodes and add it to the ML model.
Gprovince_embeddings=dif_graph_embeddings(Gprovince)
GLoc_topic_Loc_cust_embeddings=dif_graph_embeddings(GLoc_topic_Loc_cust)
GLocation_topic_Location_embeddings=dif_graph_embeddings(GLocation_topic_Location)

#export the graph embedding features
#Gprovince_embeddings.to_csv (r'C:\dev\data\interestProfile\Gprovince_embeddings.csv' ,sep=',', encoding='utf-8' )
#GLoc_topic_Loc_cust_embeddings.to_csv (r'C:\dev\data\interestProfile\GLoc_topic_Loc_cust_embeddings.csv' ,sep=',', encoding='utf-8' )
#GLocation_topic_Location_embeddings.to_csv (r'C:\dev\data\interestProfile\GLocation_topic_Location_embeddings.csv' ,sep=',', encoding='utf-8' )

#%%
#model = Node2Vec(Gprovince, dimensions=164, walk_length=2, num_walks=60, p = 1, q = 1).fit(window=10, min_count=1)#num_walks=num walk per node,  window=max distance between nodes
#print('model done')
