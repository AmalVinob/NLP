import logging 
import pandas as pd
from zenml import step
from typing import Tuple
from typing import Annotated

from sklearn.base import RegressorMixin
from source.evaluate_model import perplexity, CoherenceScore

@step
def evaluation(model:RegressorMixin, df: pd.DataFrame)-> Tuple[
    float ,
    float
]:
    """
    Evaluating the model on the ingested data

    args :
        df : preprocessed data
        model : LDA model

    return :
        EValuation scores 
    """
    try:
        coherence_class =  CoherenceScore()
        coherence_score = coherence_class.cacl_score(df=df, model= model)

        perplexity_class =  CoherenceScore()
        perplexity_score = perplexity_class.cacl_score(df=df, model= model)

        return coherence_score, perplexity_score
    
    except Exception as e:
        logging.error(f"error in evaluating the model : {e}")
        raise e

