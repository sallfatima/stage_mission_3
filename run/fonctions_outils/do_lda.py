# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Importer les packages et modules utiles
#------------------------------------------------------------------------------
import pandas as pd
import sys
import time
import scipy.sparse as sps

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from run.fonctions_outils.stopword_spec import main as stopword_spec
from run.fonctions_outils.spell_checker import main as spell_checker
import treetaggerwrapper

#------------------------------------------------------------------------------
# do_lda
# Renvoie les identifiants d'annonces répondant exactement à la requête
# input :
#### data_samples : list des textes à étudier
#### lang : langue des textes à étudier (french ou english)
#### n_features : nombre max de features
#### n_topics : nombre de topics
#### n_top_words : nombre de top_words
# output :
#### doc_topics
#### Topics
#------------------------------------------------------------------------------

def main(data_samples, lang, n_features, n_topics, n_top_words):

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")    

    # Get the top 1000 tokens in order to correct and lemmatize them
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words=stopword_spec(lang))

    t0 = time.time()
    tf = tf_vectorizer.fit_transform(data_samples)

    print("done in %0.3fs." % (time.time() - t0))
    
    # extract the top 1000 words for later use
    words_list = list(tf_vectorizer.vocabulary_.keys())
    
    print("Initialization of the spell checker on tokens...")
    
    t0 = time.time()
    
    # Check spelling of the top 1000 words
    corrected_words = spell_checker(words_list)

    print("done in %0.3fs." % (time.time() - t0))
    
    print("Initialization of the Lemmatizer...")
    
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')
    words_lemmatized = []
    for word in corrected_words:
        lemma = treetaggerwrapper.make_tags(tagger.tag_text(word),exclude_nottags=False)[0].lemma
        words_lemmatized.append(lemma)
    
    # Dict containing {unmodified words: words corrected and lemmatized}
    word_to_lemma_dict = dict(zip(words_list, words_lemmatized))
    
    # Transform the matrix to take into account spell check and lemmatization
    # 1 - Convert sparse matrice to dataframe to chan
    tf_df = pd.DataFrame(tf.A, columns=tf_vectorizer.get_feature_names())
    
    # 2 - Change the name of the columns by the corrected and lemmatized words
    tf_df.rename(index=str, columns=word_to_lemma_dict, inplace=True)

    # 3 - Groupby columns with same name and sum of counts
    tf_df = tf_df.groupby(by=tf_df.columns, axis=1).sum()
    
    # 4 - Convert df back to sparse matrix
    tf = sps.csr_matrix(tf_df)
    
    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (len(data_samples), n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time.time()
    doc_topics = lda.fit_transform(tf)
    print("done in %0.3fs." % (time.time() - t0))
    
    tf_feature_names = list(tf_df.columns)

    Topics = pd.DataFrame()

    
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

        for count, word in enumerate(topic_words):
         
            probabilite=sorted(lda.components_[topic_idx][:-n_top_words - 1:-1], reverse=True)
            total=sum(probabilite)
            #print(probabilite)
            topic_words[count] = word + " ( %0.3f )"  % (probabilite[count]/total*100) 
            
            #topic_words[count] = word + " ( %0.3f )"  % (lda.components_[topic_idx, count]/lda.components_[topic_idx][:-n_top_words - 1:-1].sum()*100) 
            #topic_words[count] = word + " (" + str(lda.components_[topic_idx, count]/lda.components_[topic_idx].sum()*100) + ")" 
			 #print(topic_words[i])
        Topics[topic_idx] = topic_words
       
    Topics = Topics.transpose()
	# frequence = pd.DataFrame(lda.components_)
	 #frequence = frequence.transpose()
    print("end LDA")
    return doc_topics, Topics



if __name__ == "__main__":
    main(sys.argv[0:])