import numpy as np
import utm
import matplotlib.pyplot as plt
import time
import queue
from datetime import datetime, timedelta
import paho.mqtt.client as mqtt
from sklearn.cluster import DBSCAN

from pymongo import MongoClient
# import boto3  # Uncomment for AWS
# from google.cloud import storage  # Uncomment for GCP
# For Azure, use the `azure.storage.blob` or similar packages
class MongoDB:
    def __init__(self, connection_string):
        self.client = MongoClient(connection_string)
        self.db = self.client['your_database']  # Replace with your database name
        self.collection = self.db['your_collection']  # Replace with your collection name

    def insert_data(self, data):
        # Insert data into the collection
        self.collection.insert_one(data)

    def retrieve_data(self, query):
        # Retrieve data from the collection
        return self.collection.find(query)
    
class CrowdData:
    def __init__(self, mode, credentials=None):
        self.mode = mode
        self.credentials = credentials
        self.queue = queue.Queue()
        self.client = None
        self.db = MongoDB('your_connection_string')

    def connect(self, host=None, port=None, topic=None):
        if self.mode == 'cloud':
         try:  
            print("Connecting to cloud services...")
            # AWS Connection
            self.client = boto3.client('s3', 
                                      aws_access_key_id=self.credentials['access_key'],
                                      aws_secret_access_key=self.credentials['secret_key'])
            
            # GCP Connection
            self.client = storage.Client.from_service_account_json(self.credentials['path_to_keyfile.json'])
            
            # Azure Connection (will require appropriate Azure SDK package)
            from azure.storage.blob import BlobServiceClient
            self.client = BlobServiceClient(account_url=self.credentials['account_url'], credential=self.credentials['credential'])
            
         except Exception as e:
             print("Error connecting to cloud services. Exiting.")
             
             return
        
        elif self.mode == 'offline':
            # Connect to MQTT server
            self.client = mqtt.Client("CrowdDataClient")
            self.client.on_message = self.on_message
            self.client.connect(host, port, 60)  # Change with your MQTT server details
            self.client.subscribe(topic)  # Change with your MQTT topic
            self.client.loop_start()

    def disconnect(self):
        if self.mode == 'offline':
            self.client.loop_stop()
            self.client.disconnect()

    def on_message(self, client, userdata, message):
        # This will be called for each message received in the MQTT topic
        self.queue.put(message.payload)

    def read_gps_data(self):
        # If you are using MQTT, this might fetch data from the queue and preprocess it
        # Otherwise, for cloud-based solutions, it would interact with cloud services to fetch the required data
        if not self.queue.empty():
            gps_data = self.queue.get()
            # Data preprocessing convert to [lat long] format drop altitude
            return gps_data
        return None
 
    def analyse_data(self, gps_data):
        # DBSCAN
        dbscan = DBSCAN(eps=0.01, min_samples=10)  # Adjust the parameters as needed
        clusters = dbscan.fit_predict(gps_data)

        # Get the current date and time
        current_datetime = datetime.now()
        date = current_datetime.date()
        time = current_datetime.time()

        # Get cluster centers and members
        
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Ignore noise
                continue
            cluster_points = gps_data[clusters == cluster_id]
            cluster_center = cluster_points.mean(axis=0).tolist()
            cluster_coords = cluster_points.tolist()

            # Structure the data
            data_entry = {
                "date": date,
                "time": time,
                "cluster_id": int(cluster_id),
                "cluster_center": cluster_center,
                "cluster_coords": cluster_coords
            }

            

        # Insert the structured data into the database
        self.db.insert_data(data_entry)


    def run(self, fetch_rate=5, retry_wait=10, max_retries=3):
        retries = 0

        while retries < max_retries:
            try:
                self.connect()

                while True:
                    gps_data = self.read_gps_data()
                    self.analyse_data(gps_data)
                    # self.plot_heatmap(heatmap_data) # Uncomment if you want to display the heatmap
                    time.sleep(fetch_rate)

            except Exception as e:
                print(f"Error: {e}. Retrying in {retry_wait} seconds...")
                retries += 1
                time.sleep(retry_wait)
            
            finally:
                self.disconnect()

        print("Max retries reached. Exiting.")
# For cloud mode
cloud_credentials = {}
crowd_data = CrowdData(mode='cloud', credentials=cloud_credentials)
crowd_data.run()

# For offline mode
crowd_data = CrowdData(mode='offline')
crowd_data.run()