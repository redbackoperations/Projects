import pandas as pd
import paho.mqtt.client as mqtt
import json
import time
import ssl  # Importing ssl certificates - ensures data transmitted is encrypted
from cryptography.fernet import Fernet #importing cryptogtaphy library

# adding an encryption and decryption key
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

class MQTTDataFrameHandler:
    def __init__(self, broker_address, topic, max_retries=3, retry_interval=5):
        self.broker_address = broker_address
        self.topic = topic
        self.client = mqtt.Client()
        self.client.on_message = self._on_message
        self.data = None
        self.error = None
        self.max_retries = max_retries
        self.retry_interval = retry_interval

    def _on_message(self, client, userdata, message):
        try:
            # Convert received message payload to DataFrame
            encrypted data = message.payload # decrypt and convert received message
            data_json = cipher_suite.decrypt(encrypted_data).decode('utf-8') # uses key to decrypt data
            data_json = message.payload.decode('utf-8')
            self.data = pd.read_json(data_json)
            # Add a timestamp column to the DataFrame
            self.data['timestamp'] = time.time()
        except Exception as e:
            self.error = str(e)
            
    def receive_data(self, timeout=10):
        retries = 0
        while retries < self.max_retries:
            try:
                self.client.connect(self.broker_address, 1883, 60)
                self.client.subscribe(self.topic)
                self.client.loop_start()
                start_time = time.time()
                while self.data is None and (time.time() - start_time) < timeout:
                    if self.error:
                        print(f"Error while receiving data: {self.error}")
                        break
                self.client.loop_stop()
                return self.data
            except Exception as e:
                print(f"Connection error: {e}. Retrying in {self.retry_interval} seconds...")
                retries += 1
                time.sleep(self.retry_interval)
        print("Max retries reached. Failed to receive data.")
        return None

    def send_data(self, df):
        retries = 0
        while retries < self.max_retries:
            try:
                # Add a timestamp column before sending
                df['timestamp'] = time.time()
                json_data = df.to_json()
                encrypted_data = cipher_suite.encrypt(json_data.encode('utf-8')) # Formats data to be encrypted
                self.client.connect(self.broker_address, 1883, 60)
                sself.client.publish(self.topic, encrypted_data) # Converts into bytes
                self.client.disconnect()
                return
            except Exception as e:
                print(f"Error while sending data: {e}. Retrying in {self.retry_interval} seconds...")
                retries += 1
                time.sleep(self.retry_interval)
        print("Max retries reached. Failed to send data.")
        
    def main():
    broker_address = "YOUR_BROKER_ADDRESS"
    topic = "test/topic"
    
    handler = MQTTDataFrameHandler(broker_address, topic)
    
    # SSL setup for client
    handler.client.tls_set(ca_certs="ca.crt", certfile="client.crt", keyfile="client.key", tls_version=ssl.PROTOCOL_TLS) # Path should be rewritten to find actual files
    handler.client.username_pw_set("client_username", "client_password")  # 'client_username' and 'cliet_password' should be replaced with client's actual username and password
    
    # Receiving data and converting it to DataFrame
    df_received = handler.receive_data()
    if df_received is not None:
        print(df_received)
        
        # Decrypts, processing the received data - if needed
        decrypted_data_json = cipher_suite.decrypt(df_received.to_json().encode('utf-8')).decode('utf-8')
        decrypted_df = pd.read_json(decrypted_data_json)
        print("Decrypted DataFrame:")
        print(decrypted_df)
        

    # Sending DataFrame data through MQTT
    df_to_send = pd.DataFrame({
        'A': [7, 8, 9],
        'B': [10, 11, 12]
    })
    handler.send_data(df_to_send)

if __name__ == "__main__":
    main()
    
