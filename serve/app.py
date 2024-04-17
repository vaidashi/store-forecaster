import datetime
import pandas as pd 
import pprint
from helpers.request import ForecastRequest, create_forecast_index
from registry.handler import MLFlowHandler
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

# Caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

# Logging
import logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)

handlers = {}
models = {}
MODEL_BASE_NAME = f"prophet-retail-forecaster-store-"

app = FastAPI()

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())
    logging.info("InMemory cache initiated")
    await get_service_handlers()
    logging.info("Updated global service handlers")

async def get_service_handlers():
    mlflow_handler = MLFlowHandler()
    global handlers
    handlers['mlflow'] = mlflow_handler
    logging.info("Retreving mlflow handler {}".format(mlflow_handler))
    return handlers
    
@app.get("/health/", status_code=200)
async def healthcheck():
    global handlers
    logging.info("Got handlers in healthcheck.")
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": handlers['mlflow'].check_mlflow_health()
        }

async def get_model(store_id: str):
    global handlers
    global models
    model_name = MODEL_BASE_NAME + F"store_id"
    
    if model_name not in models:
        models[model_name] = handlers['mlflow'].get_production_model(store_id)
    return models[model_name]

@app.post("/forecast/")
async def return_forecast(request: List[ForecastRequest]):
    forecasts = []

    for req in request:
        model = await get_model(req.store_id)
        forecast_index = create_forecast_index(req.begin_date, req.end_date)
        
        forecast_result = {}
        forecast_result['request'] = req.dict()
        model_prediction = model.predict(forecast_index)[['ds', 'yhat']]\
            .rename(columns={'ds': 'timestamp', 'yhat': 'value'})
        model_prediction['value'] = model_prediction['value'].astype(int)
        forecast_result['forecast'] = model_prediction.to_dict('records')
        forecasts.append(forecast_result)
    return forecasts