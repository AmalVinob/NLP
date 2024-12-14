import logging
import pandas as pd
import numpy as np
from zenml import step
from typing import Tuple
from sklearn.base import RegressorMixin

from source.model_dev import LDA_Model, NMF_Model, DataModeling #, BERTopic_Model


@step
def train_model(df: pd.DataFrame) -> Tuple[
    LDA_Model, np.ndarray, #BERTopic_Model
]:
    """
    Train the model on the preprocessed data
    
    Args:
        df - preporcessed dataframe
    return
        LDA model
    """
    try:
        lda = LDA_Model()
        lda_modeling = DataModeling(data=df, model=lda)
        lda_model, train_data = lda_modeling.train()

        # nmf = NMF_Model()
        # nmf_modeling = DataModeling(data=df, model=nmf)
        # nmf_model = nmf_modeling.train()

        # bert = BERTopic_Model()
        # bert_modeling = DataModeling(data=df, model=bert)
        # bert_model = bert_modeling.train()


        logging.info("Data Modeling completed")

        return lda_model, train_data #, nmf_model #, bert_model
    
    except Exception as e:
        logging.error(f"error while Modeling the data {e}")
        raise e


