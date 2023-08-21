import json
import queue
import paho.mqtt.client as mqtt
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
    
    
class DataHandler:
    def __init__(self, mode, credentials=None):
        self.mode = mode
        self.credentials = credentials
        self.queue = queue.Queue()
        self.client = None
        self.db = MongoDB('your_connection_string')
        self.data_reader = DataReader()
        self.data_parser = DataParser()
        self.data_processor = DataProcessor()

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

    def read_data(self):
        message_payload = self.data_reader.read(self.queue)
        if message_payload:
            parsed_data = self.data_parser.parse(message_payload)
            if parsed_data:
                structured_data = self.data_processor.process(parsed_data)
                self.db.insert_data(structured_data)

    
class DataParser:
    def parse(self, message_payload):
        try:
            raw_data = json.loads(message_payload)
            return {
                "user_id": raw_data.get("user_id"),
                "timestamp": raw_data.get("timestamp"),
                "latitude": raw_data.get("latitude"),
                "longitude": raw_data.get("longitude"),
                "gyroscope_x": raw_data.get("gyroscope").get("x"),
                "gyroscope_y": raw_data.get("gyroscope").get("y"),
                "gyroscope_z": raw_data.get("gyroscope").get("z"),
                "accelerometer_x": raw_data.get("accelerometer").get("x"),
                "accelerometer_y": raw_data.get("accelerometer").get("y"),
                "accelerometer_z": raw_data.get("accelerometer").get("z"),
                "heartrate": raw_data.get("heartrate"),
            }
        except json.JSONDecodeError:
            print("Error decoding message, skipping.")
            return None
        
class DataReader:
    def read(self, queue):
        return queue.get() if not queue.empty() else None

class DataProcessor:
    def process(self, parsed_data):
        return {
            "user_id": parsed_data["user_id"],
            "timestamp": parsed_data["timestamp"],
            "geo_coordinates": {
                "latitude": parsed_data["latitude"],
                "longitude": parsed_data["longitude"]
            },
            "sensor_readings": {
                "gyroscope": {
                    "x": parsed_data["gyroscope_x"],
                    "y": parsed_data["gyroscope_y"],
                    "z": parsed_data["gyroscope_z"]
                },
                "accelerometer": {
                    "x": parsed_data["accelerometer_x"],
                    "y": parsed_data["accelerometer_y"],
                    "z": parsed_data["accelerometer_z"]
                },
                "heartrate": parsed_data["heartrate"]
            }
        }

