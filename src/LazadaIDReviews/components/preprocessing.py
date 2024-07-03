from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from LazadaIDReviews import logger
from LazadaIDReviews.entity.config_entity import (DataDumpConfig, 
                                                  DataPreprocessingConfig)

import os
import joblib
import pandas as pd

class DumpData:
    def __init__(self, config: DataDumpConfig):
        self.config = config

    def dump_data(self) -> None:
        """dump the splited dataset to data training and testing
        """
        logger.info(f"Read reviews file.")
        dataset_reviews = pd.read_csv(self.config.reviews_path)
        dataset = dataset_reviews[["rating", "reviewContent"]].copy()
        dataset.dropna(inplace=True)
        
        logger.info(f"Split reviews file to data train and test.")
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["reviewContent"], 
            dataset["rating"], 
            test_size=self.config.params_test_size
        )
        
        # NOTE: improve the performance with ROS
        logger.info(f"Perform random over sampler.")
        ros = RandomOverSampler()
        X_train_ros, y_train_ros = ros.fit_resample(pd.DataFrame(X_train), pd.DataFrame(y_train))
        
        # NOTE: data save as series
        logger.info(f"Dump data train into {self.config.root_dir} directory.")
        X_train_ros["reviewContent"].to_pickle(self.config.input_train_path)
        X_test.to_pickle(self.config.input_test_path)
        
        # NOTE: data save as series
        logger.info(f"Dump data test into {self.config.root_dir} directory.")
        y_train_ros["rating"].to_pickle(self.config.output_train_path)
        y_test.to_pickle(self.config.output_test_path)

class Preprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def vectorize_data(self) -> None:
        """vectorize the splited dataset and dump vectorizer model
        """
        vectorizer = TfidfVectorizer()
        
        logger.info(f"Load data train in {self.config.input_train_path}.")
        X_train = joblib.load(self.config.input_train_path)
        
        logger.info(f"Load data test in {self.config.input_test_path}.")
        X_test = joblib.load(self.config.input_test_path)
        
        logger.info(f"Vectorize the data.")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        logger.info(f"Dump the vectorized data.")
        joblib.dump(X_train_vec, self.config.vectorized_train_path)
        joblib.dump(X_test_vec, self.config.vectorized_test_path)
        
        logger.info(f"Creating {self.config.model_dir} directory.")
        model_dir = str(self.config.model_dir)
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Save the vectorizer model.")
        joblib.dump(vectorizer, self.config.vectorizer_model_path)
