



# In[1]:


# Importer les packages et modules utiles
#------------------------------------------------------------------------------
from __future__ import division
import os
import pandas as pd
from run.fonctions_outils.do_lda import main as do_lda
from run.fonctions_outils.stats_descs import stats_descs
from run.fonctions_outils.stats_descs import stats_tags
from run.fonctions_outils.kw_tag import main as kw_tag


import numpy as np


import string
import re

#------------------------------------------------------------------------------
# Paramètre à configurer & Connexion
#------------------------------------------------------------------------------

#TODO : Faire un fichier avec les infos de paramétrage de la requête + copie de paramétrage dans fichier de sortie

#Paramètre de personnalisation des sorties
enterprise_name='EDF'

#TODO : construire root dir de manière standard :
##TODO :  root_dir  = "C:/Users/Rapabitb/Documents/PythonScripts/"
##TODO :  client_data_dir= [root_dir  ,enterprise_name,"/BW data/"]
##TODO :  comme ça on a tous une manière standard d'enregistrer nos données, il suffit de renseigner le nom du client

#Répertoire des datas

#root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/PernodRicard - DataIndia (en)/"
root_dir = "C:/Users/fatsall/Documents/Stage/EDF/data (FR)"


data_dir = [x[0] for x in os.walk(root_dir)]

#Paramètre du LDA
n_features=1000
n_topics=30
n_top_words=50

#Si on veut tagger les posts
is_tag=0

#Répertoire courant print(os.getcwd())

title = data_dir[1].split(" - ", 1)[1]
print(title)

writer = pd.ExcelWriter(os.getcwd()+'/Resultats/' + enterprise_name + '_result_' + title + '.xlsx',
                             engine='xlsxwriter',
                             options={'strings_to_urls': False})

def retraitement(doc):
    doc = ' '.join(word for word in str(doc).split(' ') if not '.' in word and not '/' in word) # strip urls
    doc = ' '.join(word for word in str(doc).split(' ') if not word.startswith('#'))  # strip hashtags
    doc = ' '.join(word for word in str(doc).split(' ') if not word.startswith('@'))  # strip mentions
    doc = np.unicode(doc).lower().replace("\r", " ").replace("\n", " ").replace("\t", '').replace("\"", " ").replace(
        "'", ' ').replace("-", " ")
    doc = doc.translate({ord(x): None for x in string.punctuation})  # strip punctuations
    doc = doc.translate({ord(x): None for x in '0123456789'})  # strip numbers
    doc = re.sub(' +', ' ', doc)  # strip double spaces
    #strip accents
    #doc = doc.translate(str.maketrans("āãàäâéèêëïîöôüûùÑŃńǸǹŇň", "aaaaaeeeeiioouuunnnnnnn"))

    return doc

df = pd.read_excel("C:/Users/fatsall/Documents/Stage/EDF/data (FR)/EDF - FR/corpus_faq_EDF_Entreprises.xlsx")


# coding: utf-8
#------------------------------------------------------------------------------


df['nb_word'] = df['fulltext'].apply(lambda x: str(x).count(" "))
df["fulltext_original"] = df["fulltext"]
df.head()

df["fulltext"]= df["fulltext"].apply(lambda x: retraitement(x))
#suppression des rt
df = df[~df.fulltext.str.startswith('rt ')]
# Suppression des posts sans texte

df=df[(df["fulltext"].notnull())|(df["fulltext"].str!='nan')]

df['longueur']=df['fulltext'].apply(lambda x:len(str(x)))
df=df[df['longueur']>6]
df= df.reset_index()
len(df.index)


print("End Cleaning file : "+str(len(df.index)))

data_samples = df['fulltext'].tolist()
lang='french'
#data_samples = df['fulltext'].str.encode('utf8').tolist()
data_samples = df['fulltext'].tolist()


#df=emoji_tag(df,writer)m

# Post tagging if choice

if is_tag==1:
    print('Start tagging post')
    list_var, df=kw_tag(df,lang)
    stats_tags(df, list_var, enterprise_name, title,writer)

# BASIC STAT
#stats

from run.fonctions_outils.do_lda import main as do_lda
# LDA
doc_topics, topics = do_lda(data_samples, lang, n_features, n_topics, n_top_words)


# In[2]:



doc_topics = pd.DataFrame(doc_topics)
df['topic'] = doc_topics.idxmax(axis=1)
topic_counts = df['topic'].value_counts()

# OUTPUT

#TODO : Rajouter pour chaque post la distribution des topics/le score par topic, plutôt que seulement un label du max de ces scores. Ca nous permettrait ensuite de calculer des distances inter-post.

topics_tot=topics.join(topic_counts)
topics_tot.to_excel(writer, sheet_name='desc_topic')



# Export des posts avec gestion de l'écriture des urls dans excel

list_var = [ "topic", 'fulltext', 'fulltext_original']

df[list_var].drop_duplicates().to_excel(writer, sheet_name='post')

writer.save()


# In[ ]:


dfs=df[list_var]


# In[ ]:


top_word=[topics_tot[0][int(i)] for i in dfs['topic'] ]
len(top_word)


# In[ ]:



dfs['top_word'] = top_word


# In[ ]:


dfs.head()


# In[ ]:


trouve=[re.search(dfs['top_word'][i]+"*",dfs['fulltext'][i], flags=0) for i in df.index]


# In[ ]:


trouve=[]
for i in df.index:
    if re.search(dfs['top_word'][i]+"*",dfs['fulltext'][i], flags=0)==None:
        trouve.append(0)
    else :
        trouve.append(1)
dfs['trouve'] = trouve


# In[ ]:


statistic=dfs.groupby(['topic', 'top_word'])['trouve'].sum()
statistic=pd.DataFrame(statistic)
topic_counts=dfs.groupby(['topic'])['topic'].count()
topic_counts.values
statistic['topic_counts']=topic_counts.values
statistic['pourcentage']=statistic['trouve']/statistic['topic_counts']*100
statistic=statistic.sort_values(by=['topic_counts','pourcentage'], ascending=False)
statistic

