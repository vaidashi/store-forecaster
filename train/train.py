import ray.data
import ray
import pandas as pd
import prophet
from prophet import Prophet
import logging
import os
import time
import mlflow
from mlflow.client import MlflowClient
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error, 
                             median_absolute_error)


def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df_store = df[(df['Store'] == store_id) & (df['Open'] == store_open)].reset_index(drop=True)
    df_store['Date'] = pd.to_datetime(df_store['Date'])
    df_store.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

    return df_store.sort_values('ds', ascending=True)

def train_test_split(df: pd.DataFrame, train_pct: float) -> 'tuple[pd.DataFrame, pd.DataFrame]':
    train_idx = int(len(df) * train_pct)
    df_train = df.copy().iloc[0 : train_idx]
    df_test = df.copy().iloc[train_idx : ]

    return df_train, df_test

def train_forecaster(
        df_train: pd.DataFrame, 
        seasonality: dict) -> prophet.forecaster.Prophet:
    
    model = Prophet(
        yearly_seasonality=seasonality['yearly'],
        weekly_seasonality=seasonality['weekly'],
        daily_seasonality=seasonality['daily'],
        interval_width=0.95
    )

    model.fit(df_train)

    return model

if __name__ == '__main__':
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
    logging.basicConfig(format = log_format, level = logging.INFO) 
    logging.info('Starting training process')

    tracking_uri = "http://0.0.0.0:5001"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri) 
    logging.info("Defined MLFlowClient and set tracking URI.")

    df = pd.read_csv('train/store_data/train.csv')

    store_ids = df['Store'].unique()
    logging.info(f"Unique store IDs: {store_ids}")

    start = time.time()
    # given dataset size, only sampling a few stores here for demonstration purposes
    for store_id in ['3', '6', '9']:
        with mlflow.start_run():
            logging.info("Started MLFlow run")
            # Transform dataset in preparation for feeding to Prophet
            df_transformed = prep_store_data(df, store_id=int(store_id))
            logging.info("Transformed data")

            model_name = f"prophet-retail-forecaster-store-{store_id}"
            mlflow.autolog()
            
            seasonality = {
                'yearly': True,
                'weekly': True,
                'daily': False
            }
            
            logging.info("Splitting data")
            # Split the data
            df_train, df_test = train_test_split(df=df_transformed, train_pct=0.75)
            logging.info("Data split")
            
            # Train the model
            logging.info("Training model")
            forecaster = train_forecaster(df_train=df_train, seasonality=seasonality)
            run_id = mlflow.active_run().info.run_id
            logging.info("Model trained")
            
            mlflow.prophet.log_model(forecaster, artifact_path="model")
            logging.info("Logged model")
            
            mlflow.log_params(seasonality)
            mlflow.log_metrics(
                {
                    'rmse': mean_squared_error(y_true=df_test['y'], y_pred=forecaster.predict(df_test)['yhat'], squared=False),
                    'mean_abs_perc_error': mean_absolute_percentage_error(y_true=df_test['y'], y_pred=forecaster.predict(df_test)['yhat']),
                    'mean_abs_error': mean_absolute_error(y_true=df_test['y'], y_pred=forecaster.predict(df_test)['yhat']),
                    'median_abs_error': median_absolute_error(y_true=df_test['y'], y_pred=forecaster.predict(df_test)['yhat'])
                }
            )
    
        # The default path where the MLflow autologging function stores the model
        artifact_path = "model"
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        
        # Register the model
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        logging.info("Model registered")

        # Transition model to production
        client.transition_model_version_stage(
            name=model_details.name,
            version=model_details.version,
            stage='production',
        )
        logging.info("Model transitioned to prod stage")
