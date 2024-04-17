# Overview
This repo was for experimenting further with MLflow for ML tracking and serving a time series forecaster with FastAPI. Only a subset of the stores from the data source (below) was leveraged in prediction given data size. 

## Data Source
`@misc{rossmann-store-sales,
    author = {FlorianKnauer, Will Cukierski},
    title = {Rossmann Store Sales},
    publisher = {Kaggle},
    year = {2015},
    url = {https://kaggle.com/competitions/rossmann-store-sales}
}`

# Running locally
Make sure conda is installed: https://docs.anaconda.com/free/miniconda/index.html

From project root, set up conda virtual environment by running:
1. `conda env create -f conda.yaml`
2. `conda activate store_forecaster`

## Set up registery
1. `sh registery/start-mlflow-server.sh`
1. See the MLflow server UI at `http://localhost:5001`

## Train Prophet model
1. `python train/train.py`

## Instructions for API setup to serve results
   1. `cd serve`
   2. `sh start-local.sh`

## Sample Post request
`POST http://127.0.0.1:8000/forecast HTTP/1.1
content-type: application/json`

```json
[{
  "store_id": "9"
}]
```
### Response
```json
[
  {
    "request": {
      "store_id": "9",
      "begin_date": null,
      "end_date": null
    },
    "forecast": [
      {
        "timestamp": "2024-04-17T15:58:05.661685",
        "value": 17094
      },
      {
        "timestamp": "2024-04-18T15:58:05.661685",
        "value": 17300
      },
      {
        "timestamp": "2024-04-19T15:58:05.661685",
        "value": 17443
      },
      {
        "timestamp": "2024-04-20T15:58:05.661685",
        "value": 16927
      },
      {
        "timestamp": "2024-04-21T15:58:05.661685",
        "value": 18618
      },
      {
        "timestamp": "2024-04-22T15:58:05.661685",
        "value": 18066
      },
      {
        "timestamp": "2024-04-23T15:58:05.661685",
        "value": 17209
      },
      {
        "timestamp": "2024-04-24T15:58:05.661685",
        "value": 17275
      }
    ]
  }
]
```

## Further work
1. Handling edge cases
1. Testing
1. CI/CD
1. misc.

