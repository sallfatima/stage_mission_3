# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Importer les packages et modules utiles
#------------------------------------------------------------------------------
from __future__ import division
import sys
import os
import pandas as pd
import numpy as np

# reference sentiment emojis/ match unicode
emojis_match = pd.read_csv(os.getcwd()+'/Resultats/Emoji_Sentiment_Data_v1.0.csv', encoding='utf-8')



# ------------------------------------------------------------------------------
# Fonction extraction emojis
# ------------------------------------------------------------------------------
def extract_emojis(full_text, emojis_match):
    # renvoie une liste séparée par des virgules d'emojis
    # Pour chaque emojis

    # mise au bon format du text => encoding
    text = np.unicode(full_text)

    cnt_emojis = 0
    list_emojis = list()
    for e in emojis_match['Emoji']:
        if text.count(e) >= 1:
            cnt_emojis = cnt_emojis + 1
            list_emojis.append(e)

    if cnt_emojis >= 1:

        str_emojis = ''
        for em in range(len(list_emojis)):
            if em == 0:
                str_emojis = list_emojis[em]
            else:
                str_emojis = str_emojis + "|" + list_emojis[em]

        return str_emojis
    else:
        return ''


def count_emojis(full_text, emojis_match):
    # renvoie une liste séparée par des virgules d'emojis
    # Pour chaque emojis

    # mise au bon format du text => encoding
    text = np.unicode(full_text)
    cnt_emojis = 0
    for e in emojis_match['Emoji']:
        if text.count(e) >= 1:
            cnt_emojis = cnt_emojis + text.count(e)

    return cnt_emojis


def count_unique_emojis(full_text, emojis_match):
    # renvoie une liste séparée par des virgules d'emojis
    # Pour chaque emojis

    # mise au bon format du text => encoding
    text = np.unicode(full_text)
    cnt_emojis = 0
    for e in emojis_match['Emoji']:
        if text.count(e) >= 1:
            cnt_emojis = cnt_emojis + 1

    return cnt_emojis

#------------------------------------------------------------------------------
# emojis
#
# input :
## df : DataFrame à tagger par les emojis
# output :
#------------------------------------------------------------------------------

def main(df,writer):
    print("extract emojis")
    # extraction des emojis de chaque text
    df["emojis"] = df['fulltext_original'].apply(lambda x: extract_emojis(x, emojis_match).replace("'", ''))

    df["cnt_emojis"] = df['fulltext_original'].apply(lambda x: count_emojis(x, emojis_match))
    df["cnt_distinct_emojis"] = df['fulltext_original'].apply(lambda x: count_unique_emojis(x, emojis_match))

    nb_emojis_tweets = df.groupby(['cnt_emojis'])['emojis'].count()
    nb_emojis_tweets.to_excel(writer, sheet_name='nb emojis')
    # duplications des lignes
    tweets_dupliq = pd.DataFrame()

    # On découpe le dataframe de 100
    tweets_ac_emojis = df[df['cnt_emojis'] > 0]

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    for i in chunker(tweets_ac_emojis, 10):
        tempEmojis = i.emojis.str.split("|").apply(pd.Series).stack()
        tempEmojis.index = tempEmojis.index.droplevel(-1)
        tempEmojis.name = 'emojis_unique'
        i["index"] = i.index

        df_tempEmojis = pd.DataFrame({"index": tempEmojis.index, "emojis_unique": tempEmojis})
        temp_tweets_dupliq = pd.merge(left=i, right=df_tempEmojis, how='inner', left_on='index', right_on='index')

        tweets_dupliq = tweets_dupliq.append(temp_tweets_dupliq)

    tweets_dupliq.reset_index()

    ## Fréquences des emojis
    freq_emojis = tweets_dupliq.groupby(['emojis_unique'], as_index=False)['index'].count()
    freq_emojis.to_excel(writer, sheet_name='freq emojis')
    freq_emojis.sort_values(by='index', ascending=False, inplace=True)

    # graph emojis
    df_graph = tweets_dupliq[tweets_dupliq['cnt_distinct_emojis'] > 1][['cnt_distinct_emojis', 'index', 'emojis_unique']]
    graph = pd.merge(left=df_graph, right=df_graph, how='inner', left_on='index', right_on='index', suffixes=('_L', '_R'))

    graph = graph[graph['emojis_unique_L'] != graph['emojis_unique_R']]

    graph_agg = graph.groupby(['emojis_unique_L', 'emojis_unique_R'], as_index=False)['index'].count()
    graph_agg.rename(columns={'index': 'nb_occ'}, inplace=True)
    graph_agg.sort_values(by='nb_occ', ascending=False, inplace=True)
    graph_agg.to_excel(writer, sheet_name='graph emojis')

    return df


if __name__ == "__main__":
    main(sys.argv[0:])