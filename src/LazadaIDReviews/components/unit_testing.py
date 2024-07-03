from LazadaIDReviews.entity.config_entity import UnitTestConfig
from LazadaIDReviews import logger
from mlflow.artifacts import download_artifacts
from mlflow import MlflowClient

import json
import requests

class UnitTesting:
    def __init__(self, config: UnitTestConfig):
        self.config = config
        self.req_body_key = None
        self.req_body = None
    
    def set_request_body(self) -> None:
        """predict the data with linear regression model

        Raises:
            client_error: error when access mlflow to get deployed model
            download_error: error when download vectorizer from mlflow artifact
        """
        try:
            logger.info("Set MLflow Client.")
            client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
            selected_model = client.get_model_version_by_alias(
                self.config.mlflow_model_name, 
                self.config.mlflow_deploy_model_alias
            )
            
            logger.info("Get the deployed model run id.")
            selected_run_id = selected_model.run_id
        except Exception as client_error:
            logger.error(client_error)
            raise client_error

        try:
            logger.info("Downloading vectorizer from MLflow's artifacts.")
            download_artifacts(
                run_id=selected_run_id,
                artifact_path=self.config.mlflow_input_example_path,
                dst_path=self.config.root_dir
            )
        except Exception as download_error:
            logger.error(download_error)
            raise download_error
        
        logger.info("Open MLflow input example.")
        f = open(f"{self.config.root_dir}/{self.config.mlflow_input_example_path}")
        input_example = json.load(f)

        # handle mlflow input example data
        data_key = input_example["columns"][0]
        data_val = input_example['data'][0][0]

        # request params
        self.req_body_key = data_key
        self.req_body = {
            data_key: data_val
        }
        
    def get_request_body_value(self) -> list:
        """get the request body data

        Returns:
            req_body: list type
        """
        logger.info("Get MLflow input example value.")
        req_body_value = self.req_body[self.req_body_key]
        return req_body_value
    
    def get_output_length(self):
        """get the output length of the predict result

        Returns:
            len_result: list type
        """
        logger.info("Get predicted result length.")
        result = requests.post(
            url=self.config.app_endpoint, 
            json=self.req_body
        )
        len_result = len(result.json())
        return len_result

    def is_output_type_list(self) -> bool:
        """check if the output file is list data type

        Returns:
            is_list: bool type
        """
        logger.info("Check is the predicted output is list.")
        result = requests.post(
            url=self.config.app_endpoint, 
            json=self.req_body
        )
        is_list = type(result.json()) is list
        return is_list

    def is_output_type_consistent(self) -> bool:
        """check if the output file have consistent
        data type inside a list

        Returns:
            bool type
        """
        logger.info("Check is each predicted output is integer")
        result = requests.post(
            url=self.config.app_endpoint, 
            json=self.req_body
        )
        for result in result.json():
            if type(result) is not int:
                return False
        return True