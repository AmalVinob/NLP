import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    Ingesting the data from the data path
    """
    def __init__(self, data_path: str):
        """
        Args :
            data_path : path to the data 
        """
        self.data_path = data_path
    
    def get_data(self):
        """
        ingest the data from the data path 
        """
        logging.info(f"ingesting the data from the data path")
        return pd.read_csv(self.data_path)



@step
def ingest_df(data_path: str)-> pd.DataFrame:
    """
    ingest the data from the data path

    Args:
        data_path : path to the data
    returns :
        a dataframe
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting the data : {e}")
        raise e