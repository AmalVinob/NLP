from zenml import pipeline
from steps.ingest import ingest_df
from steps.cleaning import clean_Df
from steps.model_train import train_model
from steps.evaluation import evaluation


@pipeline(enable_cache = False)
def train_pipeline(data_path : str):
    df = ingest_df(data_path=data_path)
    data = clean_Df(df)
    lda, train_Data = train_model(data)
    coherence_score, perplexity_score = evaluation(lda, train_Data)