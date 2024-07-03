from LazadaIDReviews import logger
from LazadaIDReviews.entity.config_entity import DataIngestionSQLConfig
from sqlalchemy import create_engine 
from tqdm import tqdm

import pandas as pd

class DataIngestionSQL:
    def __init__(self, config: DataIngestionSQLConfig):
        self.config = config

    def sql_to_csv(self) -> None:
        """get data from the SQL database
        """
        try:
            db = create_engine(self.config.source_URI)  
            conn = db.connect()

            logger.info(f"Querying reviews data from SQL Database.")
            df_reviews = pd.read_sql_table("reviews", conn)
            
            logger.info(f"Querying items data from SQL Database.")
            df_items = pd.read_sql_table("items", conn)
            
            logger.info(f"Querying category data from SQL Database.")
            df_category = pd.read_sql_table("category", conn)
            
            logger.info(f"Dump data from SQL Database to CSV.")
            df_reviews.to_csv(self.config.reviews_path, index=False)
            df_items.to_csv(self.config.items_path, index=False)
            df_category.to_csv(self.config.category_path, index=False)
                
            logger.info(f"Data dumped from SQL query into {self.config.root_dir} directory")
            conn.close()
        except Exception as e:
            conn.close()
            logger.error(e)
            raise e
