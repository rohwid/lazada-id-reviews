import os

from pathlib import Path
from LazadaIDReviews.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from LazadaIDReviews.utils.common import read_yaml, create_directories
from LazadaIDReviews.entity.config_entity import (DataIngestionSQLConfig, 
                                            DataDumpConfig,
                                            DataPreprocessingConfig,
                                            TrainingConfig,
                                            TrainEvaluationConfig)

"""NOTE: Delete or replace any function as you need
and don't forget to import each class config from
'../config/configuration.py' or
'src/MLProject/config/configuration.py'
"""

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])
    
    def get_data_ingestion_sql_config(self) -> DataIngestionSQLConfig:
        """read data ingestion config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: DataIngestionConfig type
        """
        data_ingest_config = self.config.ingest_from_sql

        create_directories([data_ingest_config.root_dir])

        config = DataIngestionSQLConfig(
            root_dir=data_ingest_config.root_dir,
            source_URI=os.environ["POSTGRES_URI"],
            reviews_table=data_ingest_config.reviews_table,
            reviews_path=Path(data_ingest_config.reviews_path),
            items_table=data_ingest_config.items_table,
            items_path=Path(data_ingest_config.items_path),
            category_table=data_ingest_config.category_table,
            category_path=Path(data_ingest_config.category_path) 
        )

        return config
    
    def get_dump_data_config(self) -> DataDumpConfig:
        """read data dump config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: PreprocessingConfig type
        """
        data_ingest_config = self.config.ingest_from_sql
        data_dump_config = self.config.dump_data
        dataset_params = self.params

        create_directories([data_dump_config.root_dir])

        config = DataDumpConfig(
            root_dir=data_dump_config.root_dir,
            reviews_path=data_ingest_config.reviews_path,
            input_train_path=data_dump_config.input_train_path,
            input_test_path=data_dump_config.input_test_path,
            output_train_path=data_dump_config.output_train_path,
            output_test_path=data_dump_config.output_test_path,
            params_test_size=dataset_params.TEST_SIZE
        )

        return config
    
    def get_preprocessing_data_config(self) -> DataPreprocessingConfig:
        """read preprocessing config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: PreprocessingConfig type
        """
        data_dump_config = self.config.dump_data
        vectorize_config = self.config.vectorize_data
        train_config = self.config.train_model

        create_directories([vectorize_config.root_dir])

        config = DataPreprocessingConfig(
            root_dir=vectorize_config.root_dir,
            input_train_path=Path(data_dump_config.input_train_path),
            input_test_path=Path(data_dump_config.input_test_path),
            vectorized_train_path=Path(vectorize_config.vectorized_train_path),
            vectorized_test_path=Path(vectorize_config.vectorized_test_path),
            model_dir=train_config.root_dir,
            vectorizer_model_path=Path(vectorize_config.vectorizer_model_path)
        )

        return config
    
    def get_training_config(self) -> TrainingConfig:
        """read training config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: TrainingConfig type
        """
        data_dump_config = self.config.dump_data
        vectorize_config = self.config.vectorize_data
        train_config = self.config.train_model
        train_params = self.params

        create_directories([train_config.root_dir])

        config = TrainingConfig(
            root_dir=train_config.root_dir,
            input_train_path=Path(data_dump_config.input_train_path),
            output_train_path=Path(data_dump_config.output_train_path),
            vectorized_train_path=Path(vectorize_config.vectorized_train_path),
            model_path=Path(train_config.model_path),
            params_max_iter=train_params.MAX_ITER,
            params_solver=train_params.SOLVER,
            params_n_jobs=train_params.N_JOBS
        )

        return config
    
    def get_train_eval_config(self) -> TrainEvaluationConfig:
        """read training evaluation config file and store as 
        config entity then apply the dataclasses
        
        Returns:
            config: TrainEvaluationConfig type
        """
        data_dump_config = self.config.dump_data
        vectorize_config = self.config.vectorize_data
        train_config = self.config.train_model
        eval_config = self.config.train_evaluation

        create_directories([eval_config.root_dir])

        config = TrainEvaluationConfig(
            root_dir=eval_config.root_dir,
            input_train_path=Path(data_dump_config.input_train_path),
            input_test_path=Path(data_dump_config.input_test_path),
            output_train_path=Path(data_dump_config.output_train_path),
            output_test_path=Path(data_dump_config.output_test_path),
            vectorized_train_path=Path(vectorize_config.vectorized_train_path),
            vectorized_test_path=Path(vectorize_config.vectorized_test_path),
            vectorizer_model_path=Path(vectorize_config.vectorizer_model_path),
            model_path=Path(train_config.model_path),
            score_path=Path(eval_config.score_path),
            mlflow_dataset_path=Path(eval_config.mlflow_dataset_path),
            mlflow_dataset_column=eval_config.mlflow_dataset_column,
            minio_endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            minio_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            minio_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            mlflow_tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            mlflow_exp_name=eval_config.mlflow_exp_name,
            mlflow_dataset_bucket=os.environ["PROJECT_BUCKET"],
            mlflow_run_name=eval_config.mlflow_run_name
        )

        return config