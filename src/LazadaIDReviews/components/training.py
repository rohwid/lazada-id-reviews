from LazadaIDReviews import logger
from LazadaIDReviews.entity.config_entity import TrainingConfig
from sklearn.linear_model import LogisticRegression

from LazadaIDReviews import logger

import joblib

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def logistic_regression(self) -> None:
        """train the data with linear regression model and dump the data
        """
        logger.info(f"Load vectorized data train from {self.config.vectorized_train_path}.")
        X_train_vec = joblib.load(self.config.vectorized_train_path)
        
        logger.info(f"Load data train output from {self.config.output_train_path}.")
        y_train = joblib.load(self.config.output_train_path)
        
        logger.info(f"Train the model.")
        model = LogisticRegression(
            solver=self.config.params_solver,
            max_iter=self.config.params_max_iter,
            n_jobs=self.config.params_n_jobs
        )
        
        model.fit(X_train_vec, y_train)
        
        logger.info(f"Dump the model.")
        joblib.dump(model, self.config.model_path)