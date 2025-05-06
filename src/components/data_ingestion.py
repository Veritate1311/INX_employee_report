import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Read the dataset, split into train/test, and save the splits."""
        logging.info("Starting data ingestion process.")
        try:
            # Load the dataset from the given CSV file
            df = pd.read_csv(
                r"C:\Users\Vasudha\INX_employee\notebook\data\INX_Future_Inc_Employee_Performance.csv")
            logging.info(
                "Dataset loaded successfully into a pandas DataFrame.")

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to artifacts (for backup/auditing)
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)
            logging.info(
                f"Raw data saved at {self.ingestion_config.raw_data_path}.")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed.")

            # Save the training and testing sets to artifacts
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)
            logging.info(
                f"Train data saved at {self.ingestion_config.train_data_path}, and test data saved at {self.ingestion_config.test_data_path}.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Run data ingestion standalone for testing
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    logging.info(
        f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")
