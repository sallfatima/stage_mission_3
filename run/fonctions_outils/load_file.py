# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Importer les packages et modules utiles
#------------------------------------------------------------------------------
from __future__ import division
import numpy as np
import pandas as pd
import glob
import os
import sys
import string
import re

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

def strip_accents(text):
    return text.translate(str.maketrans("āãàäâéèêëïîöôüûùÑŃńǸǹŇň", "aaaaaeeeeiioouuunnnnnnn"))

#------------------------------------------------------------------------------
# load_file
# Charge l'ensemble des fichiers XLSX / CSV du répertoire indiqué
# Nettoye les caractères spéciaux et des spécificités des fichiers FR
# input : Répertoire indiqué
# output : dataframe avec les informations nettoyées
#------------------------------------------------------------------------------

def main(data_dir, d):
    print("Start Load file")

    #FIXME : Modifier l'import des données avec un with open et gestion des erreurs de lecture

    #### Fichier xlsx

    all_files_xlsx = glob.glob(os.path.join(data_dir[d], "*.xlsx"))
    df_excel = pd.DataFrame()

    print('Liste des fichiers xlsx parcourus : '+str(all_files_xlsx))

    if len(all_files_xlsx) >= 1:

        df_from_each = (pd.read_excel(f,
                                      sep=';',
                                      header=0,
                                      index_col=0, encoding='utf8') for f in all_files_xlsx)

        df_excel = pd.concat(df_from_each, ignore_index=True)

        #Enlève la partie "querie" du fichier
        #skiprow=df_excel[df_excel['Unnamed: 7'].notnull()].index.min()

        #df_excel.columns=df_excel.loc[skiprow].tolist()

        #df_excel.drop(df_excel.index[range(skiprow+1)], inplace=True)

        df_excel.reset_index(inplace=True)

        col=[]
        for c in df_excel.columns :
            c=str(c).lower().replace(' ', '')

            col.append(c)

        df_excel.columns=col
    print('End load files xlsx')
    #### Fichier csv

    all_files_csv = glob.glob(os.path.join(data_dir[d], "*.csv"))
    df_csv=pd.DataFrame()

    print('Liste des fichiers csv parcourus : ' + str(all_files_csv))
    if len(all_files_csv)>=1:

        #TODO : gestion des erreur dans la lecture des csv

        df_csv_from_each = (pd.read_csv(f,sep=',', encoding='latin-1', error_bad_lines=False, index_col=False, dtype='unicode') for f in all_files_csv)

        df_csv = pd.concat(df_csv_from_each, ignore_index=True)

        df_csv.reset_index(inplace=True)

        col = []
        for c in df_csv.columns:
            c=c.lower().replace(' ', '')

            col.append(c)

        df_csv.columns = col

    print('End load files csv')
    ## Regroupement des 2 types de fichiers csv et excel

    frame=[df_excel,df_csv]

    df=pd.concat(frame)

    #Nettoyage de la mémoire
    del df_excel
    del df_csv

    print('End Load Files : '+str(len(df)))

    ##### Retraitement

   # print('Start Metric Calculation')
    print(df['fulltext'])
    df['nb_hashtag'] = df['fulltext'].apply(lambda x: str(x).count("#"))
    df['nb_mention'] = df['fulltext'].apply(lambda x: str(x).count("@"))
    df['nb_word'] = df['fulltext'].apply(lambda x: str(x).count(" "))

    print('Start Cleaning Text')

    #TODO : Supprimer les posts avec des coordonnées (num tel, adresse, email)

    #Suppression des lignes de titres restantes

    df = df[(df["queryname"] != "Query Name")]
    df = df[df["queryname"].notnull()]

    # Avant nettoyage du texte, on garde le champs original

    df["fulltext_original"] = df["fulltext"]

    # Nettoyage du texte

    df["fulltext"]= df["fulltext"].apply(lambda x: retraitement(x))
    
    #suppression des rt
    df = df[~df.fulltext.str.startswith('rt ')]

    # Suppression des posts sans texte

    df=df[(df["fulltext"].notnull())|(df["fulltext"].str!='nan')]

    df['longueur']=df['fulltext'].apply(lambda x:len(str(x)))
    df=df[df['longueur']>6]
    df=df.reset_index()
	
    list_var=["impact","queryname",'authorcountry',"gender","professions", 'interest',"pagetype", 'fulltext', 'fulltext_original', 'url', 'fullname', "author", "date",'nb_word']

    col = df.columns
    list_var.extend(col[col.str.contains('|'.join(['hashtag', 'mention']))])

    df = df[list_var]

    df.drop_duplicates(subset='fulltext', inplace=True)
    df.reset_index(inplace=True)
    
    

    print("End Cleaning file : "+str(len(df.index)))
    return df

if __name__ == "__main__":
    main(sys.argv[0:])