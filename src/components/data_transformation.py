import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    # Path to save the preprocessor object
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Creates a preprocessing pipeline for numerical and categorical features."""
        try:
            # Define numerical and categorical feature names for the HR dataset
            numerical_columns = [
                "Age", "DistanceFromHome", "EmpEducationLevel", "EmpEnvironmentSatisfaction",
                "EmpHourlyRate", "EmpJobInvolvement", "EmpJobLevel", "EmpJobSatisfaction",
                "NumCompaniesWorked", "EmpLastSalaryHikePercent", "EmpRelationshipSatisfaction",
                "TotalWorkExperienceInYears", "TrainingTimesLastYear", "EmpWorkLifeBalance",
                "ExperienceYearsAtThisCompany", "ExperienceYearsInCurrentRole",
                "YearsSinceLastPromotion", "YearsWithCurrManager"
            ]
            categorical_columns = [
                "Gender", "EducationBackground", "MaritalStatus", "EmpDepartment",
                "EmpJobRole", "BusinessTravelFrequency", "OverTime", "Attrition"
            ]

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Numerical pipeline: impute missing values then scale
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            # Categorical pipeline: impute missing, one-hot encode, then scale (without centering)
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Combine numerical and categorical pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            logging.error(f"Error creating data transformer object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Applies the preprocessing pipeline to the training and test data.
        Returns the transformed training array, testing array, and preprocessor file path.
        """
        logging.info("Initiating data transformation.")
        try:
            # Read the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test data into DataFrames.")

            # Obtain the preprocessing object (column transformer)
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "PerformanceRating"
            # target and ID columns to drop from features
            drop_columns = [target_column_name, "EmpID"]

            # Separate input features and target for training data
            X_train = train_df.drop(columns=drop_columns, errors='ignore')
            y_train = train_df[target_column_name]
            # Separate input features and target for testing data
            X_test = test_df.drop(columns=drop_columns, errors='ignore')
            y_test = test_df[target_column_name]
            logging.info(
                "Separated features and target for train and test datasets.")

            # Fit the preprocessor on the training features and transform both train and test features
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            logging.info(
                "Preprocessing pipeline applied to training and testing data.")

            # Concatenate the transformed features with the target to form final arrays
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save the preprocessor object to disk
            os.makedirs(os.path.dirname(
                self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)
            logging.info(
                f"Preprocessor object saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys)
