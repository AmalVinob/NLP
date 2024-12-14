import nltk

nltk.download('stopwords')

import logging
from abc import ABC, abstractmethod
import re
import pandas as pd
from nltk.corpus import stopwords

from sklearn.impute import SimpleImputer


stop_words_list = list(stopwords.words('english'))
stop = set(stop_words_list)


class DataStrategy(ABC):
    """
    Abstract class defining handling strategies
    """
    @abstractmethod
    def hand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class RemovePunctuationStrategy(DataStrategy):
    """
    Strategy for removing punctuation
    """
    def hand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # data['comments'] = data['comments'].apply(lambda x: re.sub(r'\W', ' ', x))
            data['comments'] = data['comments'].apply(lambda x: re.sub(r'\W', ' ', str(x)) if isinstance(x, str) else x)

            return data
        except Exception as e:
            logging.error(f"Error in RemovePunctuationStrategy: {e}")
            raise e


class RemoveNumbersStrategy(DataStrategy):
    """
    Strategy for removing numbers
    """
    def hand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # data['comments'] = data['comments'].apply(lambda x: re.sub(r'[0-9]', ' ', x))
            data['comments'] = data['comments'].apply(lambda x: re.sub(r'[0-9]', ' ', str(x)) if isinstance(x, str) else x)
            return data
        except Exception as e:
            logging.error(f"Error in RemoveNumbersStrategy: {e}")
            raise e


class RemoveSingleCharactersStrategy(DataStrategy):
    """
    Strategy for removing single characters
    """
    def hand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['comments'] = data['comments'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', str(x)) if isinstance(x, str) else x)
            return data
        except Exception as e:
            logging.error(f"Error in RemoveSingleCharactersStrategy: {e}")
            raise e


class RemoveStopwordsStrategy(DataStrategy):
    """
    Strategy for removing stopwords
    """
    def hand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['comments'] = data['comments'].apply(lambda x: " ".join([word for word in str(x).split() if word not in stop]) if isinstance(x, str) else x)
            return data
        except Exception as e:
            logging.error(f"Error in RemoveStopwordsStrategy: {e}")
            raise e


class NormalizeCaseStrategy(DataStrategy):
    """
    Strategy for normalizing text to lowercase
    """
    def hand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['comments'] = data['comments'].apply(lambda x: x.lower() if isinstance(x, str) else x)
            return data
        except Exception as e:
            logging.error(f"Error in NormalizeCaseStrategy: {e}")
            raise e


class RemoveExtraWhitespaceStrategy(DataStrategy):
    """
    Strategy for removing extra whitespace
    """
    def hand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['comments'] = data['comments'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip() if isinstance(x, str) else x)
            return data
        except Exception as e:
            logging.error(f"Error in RemoveExtraWhitespaceStrategy: {e}")
            raise e
        
class RemoveNaNStrategy(DataStrategy):
    """
    Strategy to remove NaN values from the data.
    """
    def hand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Drop rows with NaN values in the 'comments' column
            data = data.dropna(subset=['comments']).reset_index(drop=True)
            return data
        except Exception as e:
            logging.error(f"Error in RemoveNaNStrategy: {e}")
            raise e



class imputation(DataStrategy):
    """
    Stratergy for imputing nan values
    """
    def hand_data(self, data):
        try:
            imputer = SimpleImputer(strategy='most_frequent')
            data['comments'] = imputer.fit_transform(data['comments'])
        except Exception as e:
            logging.error(f"Error in imputing strategy: {e}")
            raise e


class DataCleaning:
    """
    class for cleaning the data : which removes punctations, numbers, singlecharacters, whitespaces and stopwords.
    """
    def __init__(self, data:pd.DataFrame, strategy : DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self)->pd.DataFrame:
        """
        clean Data
        """
        try:
            return self.strategy.hand_data(self.data)
        except Exception as e:
            logging.error("Error in handling data {}".format(e))
            raise e






# # Example dataset 
# data = pd.DataFrame({
#     'listing_id': [1, 2],
#     'comments': [
#         "This is a comment! It has numbers like 123 and punctuation.",
#         "Another _comment_ with single characters a b c and STOPWORDS."
#     ]
# })


# class PreprocessingPipeline:
#     """
#     Pipeline to apply multiple preprocessing strategies sequentially
#     """
#     def __init__(self, strategies):
#         self.strategies = strategies

#     def execute(self, data: pd.DataFrame) -> pd.DataFrame:
#         for strategy in self.strategies:
#             data = strategy.hand_data(data)
#         return data




# strategies = [
#     RemovePunctuationStrategy(),
#     RemoveNumbersStrategy(),
#     NormalizeCaseStrategy(),
#     RemoveSingleCharactersStrategy(),
#     RemoveStopwordsStrategy(),
#     RemoveExtraWhitespaceStrategy()
# ]


# pipeline = PreprocessingPipeline(strategies)
# processed_data = pipeline.execute(data)

# print(processed_data)
