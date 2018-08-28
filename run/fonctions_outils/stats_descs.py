# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Importer les packages et modules utiles
#------------------------------------------------------------------------------
from __future__ import division

import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#------------------------------------------------------------------------------
# split_and_stats
# fonction permettant de calculer les statistiques pour des champs fusionnés séparés par une virgule
# input :
## df : dataframe à analyser avec 2 variables : catégorie et le volume
# output : dataframe avec les stats compilées
#------------------------------------------------------------------------------
def split_and_stats (df):

    #on renomme les champs pour être générique
    df.columns = ['kw_category', 'count']

    df['virgule'] = df['kw_category'].apply(lambda x: str(x).count(','))
    file_ss = df[df['virgule'] == 0]
    #FIXME
    file_ss.loc[:, 'category_unique'] = file_ss['kw_category']
    file_ac = df[df['virgule'] > 0]

    if len(file_ac) > 0:

        tempTags = file_ac.kw_category.str.split(',').apply(pd.Series).stack()
        tempTags.index = tempTags.index.droplevel(-1)
        tempTags.name = 'category_unique'
        file_review = file_ac.join(tempTags)

        file_review = file_review.append(file_ss)
    else:
        file_review = file_ss

    file_review['category_unique'] = file_review['category_unique'].apply(lambda x: str(x).strip())

    stats = file_review.groupby(['category_unique'], as_index=False)['count'].sum()

    return stats


def cloud_hash_mention(dict,file):

    wc = WordCloud(background_color="white",  max_words=200, width=800,
                   height=400)
    wc.generate_from_frequencies(dict)

    plt.figure(figsize=(10, 20), dpi=200)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(file)

#------------------------------------------------------------------------------
# stats_desc
# Extrait les statistiques descriptives du fichier
# input :
## df : dataframe à analyser
## enterprise_name : nom de l'entreprise pour personnalisation des sorties
## title : Titre du fichier traité pour personnalisation des sorties
# output :
#------------------------------------------------------------------------------

def stats_descs(df, enterprise_name, title, writer):
    # Statistiques descriptives sur le fichier

    #TODO : Rajouter les proportions (total et fonction de NNR)
    #TODO : Rajouter la part de posts avec photo
    #TODO : Rajouter nombre de posts par personnes

    pageType = df["pagetype"].value_counts()
    pageType.sort_values(ascending=False).to_excel(writer, sheet_name='platform')

    gender = df[df["pagetype"] == "twitter"]["gender"].value_counts()
    gender.sort_values(ascending=False).to_excel(writer, sheet_name='t_gender')

    professions = df[(df["pagetype"] == "twitter")].groupby(["professions"], as_index=False)['fulltext'].count()
    professions_stat=split_and_stats(professions)
    professions_stat.sort_values(['count'],ascending=False).to_excel(writer, sheet_name='t_prof_niv1')

    professions_stat['profession_niv1']=professions_stat['category_unique'].apply(lambda x : x.split('(')[-1].replace(")",""))
    professions_stat_niv1=professions_stat.groupby(["profession_niv1"], as_index=False)['count'].sum()
    professions_stat_niv1.sort_values(['count'],ascending=False).to_excel(writer, sheet_name='t_prof_niv2')

    interest = df[(df["pagetype"] == "twitter")].groupby(["interest"], as_index=False)['fulltext'].count()
    interest_stat=split_and_stats(interest)
    interest_stat.sort_values(['count'], ascending=False).to_excel(writer, sheet_name='t_interest')

    author = df["authorcountry"].value_counts()
    author.sort_values(ascending=False).to_excel(writer, sheet_name='author_country')

    def mettre_en_entier(row):
        try:
            return int(row)
        except ValueError:
            return np.nan

    df["impact"] = df["impact"].apply(lambda row: mettre_en_entier(row))
    platform_impact = df.groupby(["pagetype"], as_index=False)['impact'].mean()
    platform_impact.sort_values(['impact'], ascending=False).to_excel(writer, sheet_name='platform_impact')

    stats_hash_mention(df, enterprise_name, title, writer)


def stats_tags(df, list_var, enterprise_name, title,writer):

    #stats par tags
    list_cat=[]
    list_val=[]
    for v in list_var:
        list_cat.append(v)
        list_val.append(len(df)-len(df[df[v]==""]))

    result_cat=pd.DataFrame({'category':list_cat, 'count':list_val})
    result_cat.sort_values(by=['count'], ascending=False).to_excel(writer, sheet_name='category')


    #stats par mots clés dans les tags
    for v in list_var :
        stat_cat=df.groupby([v],as_index=False)['fulltext'].count()
        stat_tag=split_and_stats(stat_cat)

        stat_tag.sort_values(by=['count'], ascending=False).to_excel(writer, sheet_name='tag_'+v)

def stats_hash_mention(df, enterprise_name, title,writer):
    #Stats le volume de mentions/hashtags par post

    hashtag = df["nb_hashtag"].value_counts()
    hashtag.to_excel(writer, sheet_name='nb hashtags')

    mention = df["nb_mention"].value_counts()
    mention.to_excel(writer, sheet_name='nb mentions')

    #Stats sur les hashtags et mentions utilisés.
    df_hashtag=pd.DataFrame()
    df_mention=pd.DataFrame()
    df['nb_post']=1

    for c in df.columns:
        if ('hashtag' in c) and (c!='nb_hashtag'):
            df_tmp_h=split_and_stats(df[[c, 'nb_post']])
            frame=[df_hashtag,df_tmp_h]
            df_hashtag = pd.concat(frame)

        if ('mention' in c) and (c!='nb_mention'):
            df_tmp_m=split_and_stats(df[[c, 'nb_post']])
            frame=[df_mention,df_tmp_m]
            df_mention = pd.concat(frame)

    df_hashtag_def=df_hashtag.groupby(['category_unique'],as_index=False)['count'].sum()
    df_hashtag_def.sort_values(by=['count'], ascending=False).to_excel(writer, sheet_name='type_hashtag')

    df_mention_def = df_mention.groupby(['category_unique'], as_index=False)['count'].sum()
    df_mention_def.sort_values(by=['count'], ascending=False).to_excel(writer, sheet_name='type_mention')

    # Nuage de mentions et hashtags
    keysh = list(df_hashtag_def[df_hashtag_def['category_unique']!='nan']['category_unique'])
    valuesh = list(df_hashtag_def[df_hashtag_def['category_unique']!='nan']['count'])
    dict_hash = dict(zip(keysh, valuesh))

    keysm = list(df_mention_def[df_mention_def['category_unique']!='nan']['category_unique'])
    valuesm = list(df_mention_def[df_mention_def['category_unique']!='nan']['count'])
    dict_men = dict(zip(keysm, valuesm))

    cloud_hash_mention(dict_hash, os.getcwd()+'/Resultats/' + enterprise_name + '_wordcloud_hashtags_convivialite.jpg')
    cloud_hash_mention(dict_men, os.getcwd()+'/Resultats/' + enterprise_name + '_wordcloud_mentions_convivialite.jpg')


if __name__ == "__main__":
    stats_descs(sys.argv[0:])