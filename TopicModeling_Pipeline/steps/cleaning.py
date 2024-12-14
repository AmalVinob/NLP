import logging
from zenml import step
import pandas as pd

from source.data_cleaning import DataCleaning, RemovePunctuationStrategy, RemoveNumbersStrategy, RemoveSingleCharactersStrategy, RemoveExtraWhitespaceStrategy, RemoveStopwordsStrategy, NormalizeCaseStrategy, RemoveNaNStrategy, imputation
# from typing import Annotated
# from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@step
def clean_Df(df: pd.DataFrame)-> pd.DataFrame:
    """
    Cleans the data, which removes punctations, numbers, singlecharacters, whitespaces and stopwords.
    Args:
        df : raw un processes text data
    return 
        processed Dataframe
    """
    try:
        # df = df.dropna()

        rm_nan_strategy = RemoveNaNStrategy()
        data_rm_nan = DataCleaning(data=df, strategy= rm_nan_strategy)
        cleaned_data = data_rm_nan.handle_data()


        normalize_strategy = NormalizeCaseStrategy()
        data_norm = DataCleaning(cleaned_data, normalize_strategy)
        cleaned_data = data_norm.handle_data()

        punctuation_stratergy = RemovePunctuationStrategy()
        data_rmv_puch = DataCleaning(cleaned_data, punctuation_stratergy)
        cleaned_data = data_rmv_puch.handle_data()

        number_strategy = RemoveNumbersStrategy()
        data_rmv_num = DataCleaning(cleaned_data, number_strategy)
        cleaned_data = data_rmv_num.handle_data()

        singlechar_strategy = RemoveSingleCharactersStrategy()
        data_rmv_single_char = DataCleaning(cleaned_data, singlechar_strategy)
        cleaned_data = data_rmv_single_char.handle_data()

        Whitespace_strategy = RemoveExtraWhitespaceStrategy()
        data_rmv_whitespace = DataCleaning(cleaned_data, Whitespace_strategy)
        cleaned_data = data_rmv_whitespace.handle_data()

        stopword_strategy = RemoveStopwordsStrategy()
        data_rmv_stopword = DataCleaning(cleaned_data, stopword_strategy)
        cleaned_data = data_rmv_stopword.handle_data()

        

        cleaned_data = cleaned_data.dropna()

        if cleaned_data.isnull().any().any():
            logging.error("Data contains NaN values, cannot proceed with training.")
            raise ValueError("Data contains NaN values, cannot proceed with training.")


        logging.info("data cleaning completed")

        imputation_strategy = RemoveNaNStrategy()
        data_impute = DataCleaning(data=cleaned_data, strategy= imputation_strategy)
        cleaned_data = data_impute.handle_data()
        
        return cleaned_data
    

    
    except Exception as e:
        logging.error(f"error while cleaning the data {e}")
        raise e
