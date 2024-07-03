from LazadaIDReviews.config.configuration import ConfigurationManager
from LazadaIDReviews.components.preprocessing import DumpData, Preprocessing
from LazadaIDReviews import logger

STAGE_NAME = "Preprocessing"

class PreprocessingPipeline:
    def __init__(self):
        pass

    def dump_data_pipeline(self):
        try:
            config = ConfigurationManager()
            dump_data_config = config.get_dump_data_config()
            data_ingestion = DumpData(config=dump_data_config)
            data_ingestion.dump_data()
        except Exception as e:
            logger.error(e)
            raise e
    
    def preprocessing_pipeline(self):
        try:
            config = ConfigurationManager()
            preprocessing_config = config.get_preprocessing_data_config()
            preprocessing = Preprocessing(config=preprocessing_config)
            preprocessing.vectorize_data()
        except Exception as e:
            logger.error(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f"\n\n")
        logger.info(f">>>>>>> Stage {STAGE_NAME} Started <<<<<<<")
        
        obj = PreprocessingPipeline()
        obj.dump_data_pipeline()
        obj.preprocessing_pipeline()
        
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e