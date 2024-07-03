from LazadaIDReviews import logger
from LazadaIDReviews.entity.config_entity import TrainEvaluationConfig
from LazadaIDReviews.utils.common import save_json
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.dataset_source import DatasetSource
from sklearn.metrics import classification_report

import os
import boto3
import joblib
import mlflow
import string
import random
import pandas as pd

from LazadaIDReviews import logger

class TrainEvaluation:
    def __init__(self, config: TrainEvaluationConfig):
        self.config = config

    def get_prediction(self, model, X_input_vec, X_input) -> pd.DataFrame:
        """predict the input data with the model
        
        Args:
            model (Any): the machine learning model
            X_input_vec (Any): the vectorized input data
            X_input (pd.Series): the input data
        
        Returns:
            pd.Dataframe: prediction result in dataframe
        """
        y_predict = pd.Series(model.predict(X_input_vec), index = X_input.index)
        
        return y_predict
    
    def get_report(self, y_output, y_predict) -> dict:
        """generate the classification report and dump the report as json
        
        Args:
            y_output (pd.Series): the actual output data
            y_predict (pd.Series): the prediction result
        
        Returns:
            dict: classification report in dict format
        """
        metrics = classification_report(y_output, y_predict, output_dict=True)
        
        logger.info(f"Save report as json.")
        save_json(path=self.config.score_path, data=metrics)
        
        logger.info(f"Show the training report.")
        print(f"\n{classification_report(y_output, y_predict)}")
        
        return metrics
    
    def get_mlflow_metrics(self, metrics) -> dict:
        """generate the classification report for MLflow

        Args:
            metrics (dict): the classification report
        
        Returns:
            dict: classification report in dict format
        """
        mlflow_metrics = {}

        for rating in range(len(metrics) - 3):
            data_metric = metrics[str(rating + 1)]
            for name, value in data_metric.items():
                mlflow_metrics[name + "_" + str(rating + 1)] = value
        
        return mlflow_metrics
    
    def get_dataset(self, X_input, y_output, y_predict) -> pd.DataFrame:
        """construct the dataset and save as dataframe and csv file
        
        Args:
            X_input (pd.Series): the input data
            y_output (pd.Series): the actual output data
            y_predict (pd.Series): the prediction result
        
        Returns:
            pd.Dataframe: prediction result in dataframe
        """
        train_eval_result = pd.concat([X_input, y_output, y_predict], axis = 1)
        train_eval_result.columns = self.config.mlflow_dataset_column
        train_eval_result.to_csv(self.config.mlflow_dataset_path, index=False)
        
        return train_eval_result
        
    def get_mlflow_dataset(self, mlflow_dataset, run_name) -> PandasDataset:
        """convert the dataset into MLflow's dataset format
        
        Args:
            mlflow_dataset (pd.Series): the project dataset to train and the result
            run_name (str): the name of MLflow runs
        
        Returns:
            PandasDataset: the dataset in Pandas MLflow format
        """
        mlflow_dataset: PandasDataset=mlflow.data.from_pandas(
            mlflow_dataset,
            source=DatasetSource.load(f"s3://{self.config.mlflow_dataset_bucket}/{run_name}.csv"),
            name=f"{run_name}",
            targets=self.config.mlflow_dataset_column[1],
            predictions=self.config.mlflow_dataset_column[2]
        )
        
        logger.info(f"Remove {self.config.mlflow_dataset_path} file from local.")
        os.remove(self.config.mlflow_dataset_path)
        
        return mlflow_dataset
    
    def s3_upload_mlflow_dataset(self, run_name) -> None:
        """upload the dataset into MinIO with MLflow run_name
        
        Args:
            run_name (str): the name of MLflow runs
        """
        s3 = boto3.client('s3',
                              endpoint_url=self.config.minio_endpoint_url,
                              aws_access_key_id=self.config.minio_access_key_id,
                              aws_secret_access_key=self.config.minio_secret_access_key)
        
        try:
            s3.upload_file(
                self.config.mlflow_dataset_path, 
                self.config.mlflow_dataset_bucket, 
                f'{run_name}.csv'
            )    
        except Exception as e:
            logger.error(e)
            raise e
    
    def mlflow_log_train(self) -> None:
        """perform experimentation with MLflow to evaluate the training result
        """
        logger.info(f"Load vectorized data train from {self.config.vectorized_train_path}.")
        X_train_vec = joblib.load(self.config.vectorized_train_path)
        # X_test_vec = joblib.load(self.config.vectorized_test_path)
        
        logger.info(f"Load data train from {self.config.input_train_path}.")
        X_train = joblib.load(self.config.input_train_path)
        X_test = joblib.load(self.config.input_test_path)
        
        logger.info(f"Load data train output from {self.config.output_train_path}.")
        y_train = joblib.load(self.config.output_train_path)
        # y_test = joblib.load(self.config.output_test_path)
        
        logger.info(f"Load the model.")
        model = joblib.load(self.config.model_path)
        
        logger.info(f"Predicting the data train.")
        y_train_pred = self.get_prediction(model, X_train_vec, X_train)
        
        logger.info(f"Generate classification report.")
        train_report = self.get_report(y_train, y_train_pred)
        
        logger.info(f"Set tracking URI.")
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        
        logger.info(f"Set experiment name.")
        mlflow.set_experiment(self.config.mlflow_exp_name)
        
        logger.info(f"Set run name.")
        flag = ''.join(random.choices(
            string.ascii_uppercase + string.ascii_lowercase + string.digits, 
            k=5))
        run_name = f"{self.config.mlflow_run_name}-{flag}"
        
        logger.info(f"Contruct report for MLflow.")
        mlflow_metrics = self.get_mlflow_metrics(train_report)
        
        logger.info(f"Contruct MLflow dataset file in {self.config.mlflow_dataset_path}.")
        mlflow_train_dataset = self.get_dataset(X_train, y_train, y_train_pred)

        logger.info(f"Contruct MLflow input example")
        sample = 10
        input_example = {"reviewContents": X_test.to_list()[:sample]}

        logger.info(f"Experiement tracking to evaluate model with MLflow.")
        with mlflow.start_run(run_name=run_name):
            logger.info(f"Upload {self.config.mlflow_dataset_path} file to MinIO.")
            self.s3_upload_mlflow_dataset(run_name)
            
            logger.info(f"Set MLflow dataset.")
            dataset = self.get_mlflow_dataset(mlflow_train_dataset, run_name)

            logger.info(f"Logging to MLflow as an experiment.")
            model_params = model.get_params()
            mlflow.log_params(model_params)
            mlflow.log_metrics(mlflow_metrics)
            mlflow.log_input(dataset, context="training")
            mlflow.log_artifact(self.config.vectorizer_model_path, "vectorizer")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="models",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                registered_model_name="logistic_regression",
                input_example=input_example
            )
            
            mlflow.set_tags(
                {
                    "dataset": "review contents training dataset and prediction result",
                    "model": "logistic_regression"
                }
            )