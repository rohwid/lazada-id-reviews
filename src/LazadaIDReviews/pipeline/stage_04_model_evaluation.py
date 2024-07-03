from LazadaIDReviews.config.configuration import ConfigurationManager
from LazadaIDReviews.components.model_evaluation import TrainEvaluation
from LazadaIDReviews import logger

STAGE_NAME = "Training Evaluation"

class TrainEvaluationPipeline:
    def __init__(self):
        pass

    def pipeline(self):
        try:
            config = ConfigurationManager()
            eval_config = config.get_train_eval_config()
            evaluation = TrainEvaluation(config=eval_config)
            evaluation.mlflow_log_train()
        except Exception as e:
            logger.error(e)
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f"\n\n")
        logger.info(f">>>>>>> Stage {STAGE_NAME} Started <<<<<<<")
        
        obj = TrainEvaluationPipeline()
        obj.pipeline()
        
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
