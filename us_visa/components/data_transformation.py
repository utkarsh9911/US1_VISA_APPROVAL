# import sys
# import numpy as np
# import pandas as pd
# from imblearn.combine import SMOTEENN
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
# from sklearn.compose import ColumnTransformer

# from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
# from us_visa.entity.config_entity import DataTransformationConfig
# from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
# from us_visa.exception import USvisaException
# from us_visa.logger import logging
# from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
# from us_visa.entity.estimator import TargetValueMapping


# class DataTransformation:
#     def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
#                  data_transformation_config: DataTransformationConfig,
#                  data_validation_artifact: DataValidationArtifact):
#         """
#         Initializes the DataTransformation class with artifacts and configurations.
#         """
#         try:
#             self.data_ingestion_artifact = data_ingestion_artifact
#             self.data_transformation_config = data_transformation_config
#             self.data_validation_artifact = data_validation_artifact
#             self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
#         except Exception as e:
#             raise USvisaException(e, sys)

#     @staticmethod
#     def read_data(file_path) -> pd.DataFrame:
#         """
#         Reads data from a file path and returns a pandas DataFrame.
#         """
#         try:
#             return pd.read_csv(file_path)
#         except Exception as e:
#             raise USvisaException(e, sys)

#     def get_data_transformer_object(self) -> Pipeline:
#         """
#         Creates and returns a data transformer object for preprocessing.
#         """
#         logging.info("Creating data transformer pipeline.")

#         try:
#             numeric_transformer = StandardScaler()
#             oh_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
#             ordinal_encoder = OrdinalEncoder()

#             logging.info("Initialized transformers for numeric, one-hot, and ordinal encoding.")

#             oh_columns = self._schema_config['oh_columns']
#             or_columns = self._schema_config['or_columns']
#             transform_columns = self._schema_config['transform_columns']
#             num_features = self._schema_config['num_features']

#             transform_pipe = Pipeline(steps=[
#                 ('power_transformer', PowerTransformer(method='yeo-johnson'))
#             ])

#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("OneHotEncoder", oh_transformer, oh_columns),
#                     ("OrdinalEncoder", ordinal_encoder, or_columns),
#                     ("PowerTransformer", transform_pipe, transform_columns),
#                     ("StandardScaler", numeric_transformer, num_features)
#                 ]
#             )

#             logging.info("Data transformer pipeline created successfully.")
#             return preprocessor

#         except Exception as e:
#             raise USvisaException(e, sys)

#     def initiate_data_transformation(self) -> DataTransformationArtifact:
#         """
#         Initiates the data transformation process.
#         """
#         try:
#             if not self.data_validation_artifact.validation_status:
#                 raise Exception(self.data_validation_artifact.message)

#             logging.info("Starting data transformation.")

#             preprocessor = self.get_data_transformer_object()
#             train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
#             test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

#             input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_train_df = train_df[TARGET_COLUMN]

#             input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']

#             drop_cols = self._schema_config['drop_columns']
#             input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
#             target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())

#             input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_test_df = test_df[TARGET_COLUMN]

#             input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
#             input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)
#             target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())

#             logging.info("Applying preprocessing to train and test datasets.")

#             input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessor.transform(input_feature_test_df)

#             logging.info("Applying SMOTEENN for handling class imbalance.")

#             smt = SMOTEENN(sampling_strategy="minority")

#             input_feature_train_final, target_feature_train_final = smt.fit_resample(
#                 input_feature_train_arr, target_feature_train_df
#             )

#             input_feature_test_final, target_feature_test_final = smt.fit_resample(
#                 input_feature_test_arr, target_feature_test_df
#             )

#             train_arr = np.c_[input_feature_train_final, target_feature_train_final]
#             test_arr = np.c_[input_feature_test_final, target_feature_test_final]

#             save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
#             save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
#             save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

#             logging.info("Data transformation completed successfully.")

#             return DataTransformationArtifact(
#                 transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                 transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
#                 transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
#             )

#         except Exception as e:
#             raise USvisaException(e, sys)


# import sys

# import numpy as np
# import pandas as pd
# from imblearn.combine import SMOTEENN
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
# from sklearn.compose import ColumnTransformer

# from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
# from us_visa.entity.config_entity import DataTransformationConfig
# from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
# from us_visa.exception import USvisaException
# from us_visa.logger import logging
# from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
# from us_visa.entity.estimator import TargetValueMapping



# class DataTransformation:
#     def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
#                  data_transformation_config: DataTransformationConfig,
#                  data_validation_artifact: DataValidationArtifact):
#         """
#         :param data_ingestion_artifact: Output reference of data ingestion artifact stage
#         :param data_transformation_config: configuration for data transformation
#         """
#         try:
#             self.data_ingestion_artifact = data_ingestion_artifact
#             self.data_transformation_config = data_transformation_config
#             self.data_validation_artifact = data_validation_artifact
#             self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
#         except Exception as e:
#             raise USvisaException(e, sys)

#     @staticmethod
#     def read_data(file_path) -> pd.DataFrame:
#         try:
#             return pd.read_csv(file_path)
#         except Exception as e:
#             raise USvisaException(e, sys)

    
#     def get_data_transformer_object(self) -> Pipeline:
#         """
#         Method Name :   get_data_transformer_object
#         Description :   This method creates and returns a data transformer object for the data
        
#         Output      :   data transformer object is created and returned 
#         On Failure  :   Write an exception log and then raise an exception
#         """
#         logging.info(
#             "Entered get_data_transformer_object method of DataTransformation class"
#         )

#         try:
#             logging.info("Got numerical cols from schema config")

#             numeric_transformer = StandardScaler()
#             oh_transformer = OneHotEncoder()
#             ordinal_encoder = OrdinalEncoder()

#             logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

#             oh_columns = self._schema_config['oh_columns']
#             or_columns = self._schema_config['or_columns']
#             transform_columns = self._schema_config['transform_columns']
#             num_features = self._schema_config['num_features']

#             logging.info("Initialize PowerTransformer")

#             transform_pipe = Pipeline(steps=[
#                 ('transformer', PowerTransformer(method='yeo-johnson'))
#             ])
#             preprocessor = ColumnTransformer(
#                 [
#                     ("OneHotEncoder", oh_transformer, oh_columns),
#                     ("Ordinal_Encoder", ordinal_encoder, or_columns),
#                     ("Transformer", transform_pipe, transform_columns),
#                     ("StandardScaler", numeric_transformer, num_features)
#                 ]
#             )

#             logging.info("Created preprocessor object from ColumnTransformer")

#             logging.info(
#                 "Exited get_data_transformer_object method of DataTransformation class"
#             )
#             return preprocessor

#         except Exception as e:
#             raise USvisaException(e, sys) from e

#     def initiate_data_transformation(self, ) -> DataTransformationArtifact:
#         """
#         Method Name :   initiate_data_transformation
#         Description :   This method initiates the data transformation component for the pipeline 
        
#         Output      :   data transformer steps are performed and preprocessor object is created  
#         On Failure  :   Write an exception log and then raise an exception
#         """
#         try:
#             if self.data_validation_artifact.validation_status:
#                 logging.info("Starting data transformation")
#                 preprocessor = self.get_data_transformer_object()
#                 logging.info("Got the preprocessor object")

#                 train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
#                 test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

#                 input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
#                 target_feature_train_df = train_df[TARGET_COLUMN]

#                 logging.info("Got train features and test features of Training dataset")

#                 input_feature_train_df['company_age'] = CURRENT_YEAR-input_feature_train_df['yr_of_estab']

#                 logging.info("Added company_age column to the Training dataset")

#                 drop_cols = self._schema_config['drop_columns']

#                 logging.info("drop the columns in drop_cols of Training dataset")

#                 input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)
                
#                 target_feature_train_df = target_feature_train_df.replace(
#                     TargetValueMapping()._asdict()
#                 )


#                 input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

#                 target_feature_test_df = test_df[TARGET_COLUMN]


#                 input_feature_test_df['company_age'] = CURRENT_YEAR-input_feature_test_df['yr_of_estab']

#                 logging.info("Added company_age column to the Test dataset")

#                 input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

#                 logging.info("drop the columns in drop_cols of Test dataset")

#                 target_feature_test_df = target_feature_test_df.replace(
#                 TargetValueMapping()._asdict()
#                 )

#                 logging.info("Got train features and test features of Testing dataset")

#                 logging.info(
#                     "Applying preprocessing object on training dataframe and testing dataframe"
#                 )

#                 input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

#                 logging.info(
#                     "Used the preprocessor object to fit transform the train features"
#                 )

#                 input_feature_test_arr = preprocessor.transform(input_feature_test_df)

#                 logging.info("Used the preprocessor object to transform the test features")

#                 logging.info("Applying SMOTEENN on Training dataset")

#                 smt = SMOTEENN(sampling_strategy="minority")

#                 input_feature_train_final, target_feature_train_final = smt.fit_resample(
#                     input_feature_train_arr, target_feature_train_df
#                 )

#                 logging.info("Applied SMOTEENN on training dataset")

#                 logging.info("Applying SMOTEENN on testing dataset")

#                 input_feature_test_final, target_feature_test_final = smt.fit_resample(
#                     input_feature_test_arr, target_feature_test_df
#                 )

#                 logging.info("Applied SMOTEENN on testing dataset")

#                 logging.info("Created train array and test array")

#                 train_arr = np.c_[
#                     input_feature_train_final, np.array(target_feature_train_final)
#                 ]

#                 test_arr = np.c_[
#                     input_feature_test_final, np.array(target_feature_test_final)
#                 ]

#                 save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
#                 save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
#                 save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

#                 logging.info("Saved the preprocessor object")

#                 logging.info(
#                     "Exited initiate_data_transformation method of Data_Transformation class"
#                 )

#                 data_transformation_artifact = DataTransformationArtifact(
#                     transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                     transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
#                     transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
#                 )
#                 return data_transformation_artifact
#             else:
#                 raise Exception(self.data_validation_artifact.message)

#         except Exception as e:
#             raise USvisaException(e, sys) from e






# import sys
# import numpy as np
# import pandas as pd
# from imblearn.combine import SMOTEENN
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
# from sklearn.compose import ColumnTransformer

# from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
# from us_visa.entity.config_entity import DataTransformationConfig
# from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
# from us_visa.exception import USvisaException
# from us_visa.logger import logging
# from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
# from us_visa.entity.estimator import TargetValueMapping


# class DataTransformation:
#     def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
#                  data_transformation_config: DataTransformationConfig,
#                  data_validation_artifact: DataValidationArtifact):
#         try:
#             self.data_ingestion_artifact = data_ingestion_artifact
#             self.data_transformation_config = data_transformation_config
#             self.data_validation_artifact = data_validation_artifact
#             self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
#         except Exception as e:
#             raise USvisaException(e, sys)

#     @staticmethod
#     def read_data(file_path: str) -> pd.DataFrame:
#         try:
#             return pd.read_csv(file_path)
#         except Exception as e:
#             raise USvisaException(e, sys)

#     def validate_columns(self, df: pd.DataFrame, required_columns: list):
#         """
#         Validate that all required columns exist in the DataFrame.
#         """
#         missing_columns = [col for col in required_columns if col not in df.columns]
#         if missing_columns:
#             raise KeyError(f"Missing columns in the dataset: {missing_columns}")

#     def get_data_transformer_object(self) -> ColumnTransformer:
#         """
#         Create and return a data transformer object.
#         """
#         try:
#             logging.info("Initializing data transformers")

#             # Load column information from schema
#             oh_columns = self._schema_config['oh_columns']
#             or_columns = self._schema_config['or_columns']
#             transform_columns = self._schema_config['transform_columns']
#             num_features = self._schema_config['num_features']

#             # Define transformers
#             numeric_transformer = StandardScaler()
#             oh_transformer = OneHotEncoder()
#             ordinal_encoder = OrdinalEncoder()
#             transform_pipe = Pipeline(steps=[('transformer', PowerTransformer(method='yeo-johnson'))])

#             preprocessor = ColumnTransformer(
#                 [
#                     ("OneHotEncoder", oh_transformer, oh_columns),
#                     ("Ordinal_Encoder", ordinal_encoder, or_columns),
#                     ("Transformer", transform_pipe, transform_columns),
#                     ("StandardScaler", numeric_transformer, num_features)
#                 ]
#             )
#             logging.info("Data transformer object created successfully")
#             return preprocessor
#         except Exception as e:
#             raise USvisaException(e, sys)

#     def initiate_data_transformation(self) -> DataTransformationArtifact:
#         """
#         Perform data transformation and save transformed data.
#         """
#         try:
#             if not self.data_validation_artifact.validation_status:
#                 raise Exception(self.data_validation_artifact.message)

#             logging.info("Starting data transformation")

#             # Read input data
#             train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
#             test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

#             # Extract input and target features
#             input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_train_df = train_df[TARGET_COLUMN]

#             input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_test_df = test_df[TARGET_COLUMN]

#             # Validate required columns
#             all_columns = self._schema_config['oh_columns'] + \
#                           self._schema_config['or_columns'] + \
#                           self._schema_config['transform_columns'] + \
#                           self._schema_config['num_features']
#             self.validate_columns(input_feature_train_df, all_columns)

#             # Add derived columns
#             input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
#             input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']

#             # Drop unnecessary columns
#             drop_cols = self._schema_config['drop_columns']
#             input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
#             input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)

#             # Map target values
#             target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())
#             target_feature_test_df = target_feature_test_df.replace(


import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.estimator import TargetValueMapping



class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Training dataset")

                input_feature_train_df['company_age'] = CURRENT_YEAR-input_feature_train_df['yr_of_estab']

                logging.info("Added company_age column to the Training dataset")

                drop_cols = self._schema_config['drop_columns']

                logging.info("drop the columns in drop_cols of Training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)
                
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )


                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

                target_feature_test_df = test_df[TARGET_COLUMN]


                input_feature_test_df['company_age'] = CURRENT_YEAR-input_feature_test_df['yr_of_estab']

                logging.info("Added company_age column to the Test dataset")

                input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

                logging.info("drop the columns in drop_cols of Test dataset")

                target_feature_test_df = target_feature_test_df.replace(
                TargetValueMapping()._asdict()
                )

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Applying SMOTEENN on Training dataset")

                smt = SMOTEENN(sampling_strategy="minority")

                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )

                logging.info("Applied SMOTEENN on training dataset")

                logging.info("Applying SMOTEENN on testing dataset")

                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                logging.info("Applied SMOTEENN on testing dataset")

                logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]

                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USvisaException(e, sys) from e