import logging
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic

nltk.download('stopwords')
stopwords_list = list(stopwords.words("english"))
stop = set(stopwords_list)

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, data):
        """
        train the model
        args:
            data : preprocessed dataframe
        return :none
        """
        pass


class LDA_Model(Model):
    """
    LatentDirichletAllocation model
    """
    def __init__(self, num_topic: int = 10):
        self.num_topic = num_topic

    def train(self, data: pd.DataFrame, **kwargs):
        """
        Train the model
        Args:
            data : preprocessed dataframe
        Returns : trained LDA model and training data
        """
        try:
            # Convert data to list of comments if not already
            data = data.dropna()
            
            if isinstance(data['comments'], pd.Series):
                documents = data['comments'].tolist()
            else:
                documents = data['comments']

            # Feature extraction
            tfvec = CountVectorizer(stop_words=stopwords_list)
            train_data = tfvec.fit_transform(documents)
            tf_feat_name = tfvec.get_feature_names_out()

            # Model
            lda_model = LatentDirichletAllocation(n_components=self.num_topic)
            lda_matrix = lda_model.fit_transform(train_data)

            # Display LDA components
            lda_components = lda_model.components_

            for index, component in enumerate(lda_components):
                top_terms_key = sorted(zip(tf_feat_name, component), key=lambda t: t[1], reverse=True)
                top_terms_list = [term for term, _ in top_terms_key[:10]]  # Show top 10 terms per topic
                print(f"Topic {index}: ", top_terms_list)

            logging.info("Topic modeling complete")
            return lda_model, train_data

        except Exception as e:
            logging.error(f"Error in topic modeling: {e}")
            raise e


# class LDA_Model(Model):
#     """
#     LatentDirichletAllocation model
#     """

#     def __init__(self, num_topic:int = 10):
#         self.num_topic = 10

#     def train(self, data: pd.DataFrame, **kwargs):
#         """
#         Train the model
#         args:
#             data : preprocessed dataframe
#         return :none
#         """
#         try :
#             # feature extractoin 

#             tfvec = CountVectorizer(stop_words=stopwords_list)
#             train_data = tfvec.fit_transform(data['comments'])
#             tf_feat_name = tfvec.get_feature_names_out()


#             #model

#             lda_model = LatentDirichletAllocation(n_components=self.num_topic)
#             lda_matrix = lda_model.fit_transform(train_data)  # we need to tell to process comment

#             #display lda componets
#             lda_components = lda_model.components_
#             term = tf_feat_name


#             for index, component in enumerate(lda_components):
#                 top_terms_key = sorted(zip(term, component), key= lambda t : t[1], reverse=True)
#                 top_terms_list = list(dict(top_terms_key).keys())
#                 print(f"Topic {index}: ", top_terms_list)
            
#             logging.info("Topic modeling complete")
#             return lda_model, train_data
        
#         except Exception as e:
#             logging.error(f"Error in topic modeling: {e}")
#             raise e


class NMF_Model(Model):
    """
    Non-Negative Matrix Factorization (NMF) model
    """

    def __init__(self, num_topics: int = 10):
        self.num_topics = num_topics

    def train(self, data: pd.DataFrame, **kwargs):
        """
        Train the model
        args:
            data : preprocessed dataframe
        return : trained NMF model
        """
        try:
            # Feature extraction
            data = data.dropna()
            tfvec = CountVectorizer(stop_words=stopwords_list)
            train_data = tfvec.fit_transform(data['comments'])
            tf_feat_name = tfvec.get_feature_names_out()

            # Model
            nmf_model = NMF(n_components=self.num_topics, random_state=42)
            nmf_matrix = nmf_model.fit_transform(train_data)

            # Display NMF components
            nmf_components = nmf_model.components_
            term = tf_feat_name

            for index, component in enumerate(nmf_components):
                top_terms_key = sorted(zip(term, component), key=lambda t: t[1], reverse=True)
                top_terms_list = list(dict(top_terms_key).keys())
                print(f"Topic {index}: ", top_terms_list)

            logging.info("NMF topic modeling complete")
            return nmf_model

        except Exception as e:
            logging.error(f"Error in NMF topic modeling: {e}")
            raise e




class BERTopic_Model(Model):
    """
    BERTopic model
    """

    def __init__(self):
        self.model = BERTopic()

    def train(self, data: pd.DataFrame, **kwargs):
        """
        Train the model
        args:
            data : preprocessed dataframe
        return : trained BERTopic model
        """
        try:
            # Preprocess the data into a list of documents
            documents = data['comments'].tolist()

            # Model
            topic_model = self.model
            topics, probs = topic_model.fit_transform(documents)

            # Display top topics
            topic_info = topic_model.get_topic_info()
            print("Top topics:\n", topic_info)

            logging.info("BERTopic modeling complete")
            return topic_model

        except Exception as e:
            logging.error(f"Error in BERTopic modeling: {e}")
            raise e


class DataModeling:
    """
    For modeling the data 
    """
    def __init__(self, data :pd.DataFrame ,model):
        self.data = data
        self.model = model
    
    def train(self):
        """
        model train
        """
        try:
            return self.model.train(self.data)
        except Exception as e:
            logging.error("Error in Modeling data {}".format(e))
            raise e
