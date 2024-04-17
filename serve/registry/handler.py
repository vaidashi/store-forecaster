import mlflow
from mlflow.client import MlflowClient
from mlflow.pyfunc import PyFuncModel
from pprint import pprint

class MLFlowHandler:
    def __init__(self) -> None:
        tracking_uri = "http://0.0.0.0:5001"
        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def check_mlflow_health(self) -> None:
        try:
            experiments = self.client.search_experiments()   
            for rm in experiments:
                pprint(dict(rm), indent=4)
                return 'Service returning experiments'
        except:
            return 'Error calling MLFlow'

    def get_production_model(self, store_id: str) -> PyFuncModel:
        model_name = f"prophet-retail-forecaster-store-{store_id}"
        latest_versions_metadata = self.client.get_latest_versions(
            name=model_name
        )
        
        # latest_model_version_metadata = self.client.get_model_version(
        #     name=model_name,
        #     version=latest_versions_metadata.version
        # )
        # latest_model_run_id = latest_model_version_metadata.run_id
        # logged_model = f'runs:/{latest_model_run_id}/model'
       
        logged_model = f"runs:/c218f79684834c218eaa3e3e6d1001ed/model"
        # model = mlflow.pyfunc.load_model(logged_model)

        model = mlflow.pyfunc.load_model(model_uri = f"models:/{model_name}/production")
        return model