# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Importer les packages et modules utiles
#------------------------------------------------------------------------------
from __future__ import division
import sys
import os
import pandas as pd
from run.fonctions_outils.load_file import strip_accents
print("READ TAGGAGE")
print(os.getcwd()+'/Resultats/reference_taggage.xlsx')
ref = pd.read_excel(os.getcwd()+'/Resultats/reference_taggage.xlsx', sheet='ref')

#------------------------------------------------------------------------------
# taggage
# Permet de tagger chaque post avec des mots clés et affectation avec des catégories
# input : (lang)
## doc : post à tagger
## lang : langue du post
# output : les tags séparés par des virgules
#------------------------------------------------------------------------------

def taggage(text,type,lang):
    list_str = []
    ct = 0
    for kw in ref[((ref["langue"] == lang) | (ref["langue"].isnull())) & (ref["category"] == type)]['keyword'].unique():
        if kw in text:
            list_str.append(kw)
            ct += 1

    if ct == 0:
        return ""
    else:
        if ct == 1:
            return str(list_str[0])
        else:
            return ','.join(list_str)


#------------------------------------------------------------------------------
# KW_tag
# Permet de tagger l'ensemble du DataFrame avec des mots clés
# et affectation avec des catégories
# input :
## df : DataFrame à tagger
## lang : langue du DataFrame
# output : la liste des champs créés et le nouveau DataFrame
#------------------------------------------------------------------------------

def main(df, lang):
        
    # Copie pour conserver les accents pour le correcteur orthographique en aval
    df["fulltext_copy"] = df["fulltext"]
    # Suppresion des accents sur la copie
    df["fulltext_copy"] = df["fulltext_copy"].apply(lambda x: strip_accents(x))
    
    # Taggage du fichier avec le type d'alcool & marque & moment
    list_var=[]
    for type in ref[(ref["langue"] == lang) | (ref["langue"].isnull())].category.unique():
        print("Tagging category : "+str(type))
        df[type]=df["fulltext_copy"].apply(lambda x: taggage(x,type,lang))
        list_var.append(type)

    # Suppresion de la copie
    df.drop(["fulltext_copy"], inplace=True, axis=1)
    
    return list_var, df


if __name__ == "__main__":
    main(sys.argv[0:])