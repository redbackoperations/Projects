from airflow import DAG
from airflow.providers.microsoft.azure.transfers.local_to_wasb import LocalFilesystemToWasbOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'your_name',
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
}

dag = DAG(
    'csv_upload_to_azure',
    default_args=default_args,
    description='Upload CSV to Azure and notify on completion or failure',
    schedule_interval=None,  # Set your desired schedule_interval
    catchup=False,  # Set to False if you don't want historical runs to execute
    max_active_runs=1,
)

upload_csv_task = LocalFilesystemToWasbOperator(
    task_id='upload_csv_to_azure',
    file_path='/home/adhish/airflow/dags/dag.py',
    container_name='csv',
    blob_name='Python2.csv',
    dag=dag,
    wasb_conn_id='azur_blob_storage'
)

send_email_task = EmailOperator(
    task_id='send_email',
    to='adhishanand9@gmail.com',
    subject='CSV Upload Complete',
    html_content='The CSV file has been successfully uploaded to Azure Blob Storage.',
    dag=dag,
#    conn_id='adhishanand19'
)

send_email_task >> upload_csv_task