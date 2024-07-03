from LazadaIDReviews import logger
from LazadaIDReviews.config.configuration import ConfigurationManager
from LazadaIDReviews.components.data_ingestion import DataIngestionSQL

STAGE_NAME = "Data Ingestion"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def pipeline(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_sql_config()
            data_ingestion = DataIngestionSQL(config=data_ingestion_config)
            data_ingestion.sql_to_csv()
        except Exception as e:
            logger.error(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f"\n\n")
        logger.info(f">>>>>>> Stage {STAGE_NAME} Started <<<<<<<")
        
        obj = DataIngestionPipeline()
        obj.pipeline()
        
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e