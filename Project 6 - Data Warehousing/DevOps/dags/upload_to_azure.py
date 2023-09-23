import logging
from airflow import DAG
from datetime import datetime, timedelta
from airflow.providers.microsoft.azure.hooks.wasb import WasbHook
from airflow.operators.python import PythonOperator
from airflow.providers.microsoft.azure.transfers.local_to_wasb import LocalFilesystemToWasbOperator

# Change these to your identifiers, if needed.
AZURE_CONN_ID = "azure_connection"

default_args = {
        'owner': 'compamy_name',
        'start_date': datetime.today(),
        'retries': 3,
        'retry_delay': timedelta(minutes=5)
}

# This method is not being used at the moment. It can be used if using PythonOperator and provide a callable
def az_upload(json_file_path):
  az_hook = WasbHook.get_hook(AZURE_CONN_ID)
  logging.info("Exporting JSON to Azure Blob")
  az_hook.load_file(json_file_path, container_name="videos", blob_name="blob_name")

with DAG(
    default_args=default_args,
    dag_id="azure_upload",
    schedule_interval='* * * * *',
    catchup=False,
    tags=['example, localtoazure','azure']
) as dag:

  t1 = LocalFilesystemToWasbOperator(
    file_path="path/to/json/file/on/local/machine",
    container_name="container_name",
    blob_name="blob_name",
    wasb_conn_id=AZURE_CONN_ID,
    task_id="azure_upload_task"
    )

  t1