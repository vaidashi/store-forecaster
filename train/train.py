import ray.data
import ray
import pandas as pd
from prophet import Prophet
import logging
import os
import time
# Logging
import logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)

# todo add docstring autogen builder!!!!!!!!!!
def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df_store = df[(df['Store'] == store_id) & (df['Open'] == store_open)].reset_index(drop=True)
    df_store['Date'] = pd.to_datetime(df_store['Date'])
    df_store.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

    return df_store.sort_values('ds', ascending=True)

def train_predict(
        df: pd.DataFrame, 
        train_pct: float, 
        seasonality: dict) -> 'tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]':
    
    train_idx = int(len(df) * train_pct)
    df_train = df.copy().iloc[0 : train_idx]
    df_test = df.copy().iloc[train_idx : ]

    model = Prophet(
        yearly_seasonality=seasonality['yearly'],
        weekly_seasonality=seasonality['weekly'],
        daily_seasonality=seasonality['daily'],
        interval_width=0.95
    )

    model.fit(df_train)
    pred = model.predict(df_test)

    return pred, df_train, df_test, train_idx

@ray.remote(num_returns = 4)
def prep_train_predict(
        df: pd.DataFrame, 
        store_id: int, 
        store_open: int = 1,
        train_pct: float = .8, 
        seasonality: dict={'yearly': True, 'weekly': True, 'daily': False}
        ) -> 'tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]':
    df_store = prep_store_data(df, store_id = store_id, store_open = store_open)
    return train_predict(df_store, train_pct, seasonality)

if __name__ == '__main__':
    logging.info('Starting training process')

    df = pd.read_csv('train/store_data/train.csv')
    store_ids = df['Store'].unique()
    seasonality = {'yearly': True, 'weekly': True, 'daily': False}

    ray.init(num_cpus = 4)
    df_id = ray.put(df)

    start = time.time()
    pred_refs, train_refs, test_refs, train_idx_refs = map(
        list,
        zip(*([prep_train_predict.remote(df_id, store_id) for store_id in store_ids])),
    )
    
    ray_results = {
        'preds': ray.get(pred_refs),
        'train': ray.get(train_refs),
        'test': ray.get(test_refs),
        'train_idx': ray.get(train_idx_refs)
    }

    ray_core_time = time.time() - start
    print(f'Time elapsed: {ray_core_time - start}')
    ray.shutdown()
    logging.info('Finished training process')
