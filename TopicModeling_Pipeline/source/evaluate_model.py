import logging
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated


class Evaluation(ABC):
    """
    abstract class defining stratergy for evaluating the models
    """
    @abstractmethod
    def cacl_score(self, df : pd.DataFrame, model: RegressorMixin):
        """
        Evaluating the model on the ingested data

        args :
            df : preprocessed data
            model : LDA model

        return :
            EValuation scores 
        """
        pass


class CoherenceScore(Evaluation):
    """
    Evaluating the model using coherence score
    """

    def cacl_score(self, df: pd.DataFrame, model:RegressorMixin):
        try:
            logging.info("calculating coherence score : ")
            coherence_score = model.get_coherence_score(df['comment'].tolist())
            logging.info("cohherence score ", coherence_score)
            return coherence_score
        except Exception as e:
            logging.error("error in calculating the Coherence score : {e}".format(e))
            raise e

class perplexity(Evaluation):
    """
    Evaluating the model using perplexity
    """

    def cacl_score(self, df : pd.DataFrame, model:RegressorMixin):
        try:
            if hasattr(model, 'perplexity'):
                logging.info("calculating perplexity score : ")
                perplexity = model.perplexity(df['comments'])
                logging.info(f"perplexity score : {perplexity}")
            return perplexity
        except Exception as e:
            logging.error(f"error in calculating perplexity score {e}")
            raise e