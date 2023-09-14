from flask import Flask, request

app = Flask(__name__)

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.json
    handler = DataHandler()
    handler.save_data(data)
    print("data saved")
    return "Data received", 200




if __name__ == '__main__':
    app.run(port=5000)


import sqlite3

class DataHandler:
    def __init__(self):
        self.conn = sqlite3.connect('data.db')
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS user_data (
                    Time TEXT,
                    Agent_ID TEXT,
                    Device_Type TEXT,
                    Accelerometer_X TEXT,
                    Accelerometer_Y TEXT,
                    Accelerometer_Z TEXT,
                    Gyroscope_X TEXT,
                    Gyroscope_Y TEXT,
                    Gyroscope_Z TEXT,
                    Longitude_Degrees TEXT,
                    Longitude_Minutes TEXT,
                    Latitude_Degrees TEXT,
                    Latitude_Minutes TEXT,
                    Altitude TEXT
                )
            ''')

    def save_data(self, data):
        with self.conn:
            self.conn.execute('''
                INSERT INTO user_data (
                    Time, Agent_ID, Device_Type, 
                    Accelerometer_X, Accelerometer_Y, Accelerometer_Z, 
                    Gyroscope_X, Gyroscope_Y, Gyroscope_Z, 
                    Longitude_Degrees, Longitude_Minutes, 
                    Latitude_Degrees, Latitude_Minutes, Altitude
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['Time'], data['Agent ID'], data['Device Type'],
                data['Accelerometer X'], data['Accelerometer Y'], data['Accelerometer Z'],
                data['Gyroscope X'], data['Gyroscope Y'], data['Gyroscope Z'],
                data['Longitude Degrees'], data['Longitude Minutes'],
                data['Latitude Degrees'], data['Latitude Minutes'], data['Altitude']
            ))

    def get_all_data_for_user(self, user_id, minimum_entries=500):
        with self.conn:
            cursor = self.conn.execute('SELECT * FROM user_data WHERE Agent_ID = ?', (user_id,))
            data = cursor.fetchall()
            if len(data) < minimum_entries:
                return None  # or return an empty list
            return data

    def get_latest_data_for_user(self, user_id):
        with self.conn:
            cursor = self.conn.execute('SELECT * FROM user_data WHERE Agent_ID = ? ORDER BY Time DESC LIMIT 1', (user_id,))
            return cursor.fetchone()

    def get_all_user_coordinates(self,user_id):
        with self.conn:
            cursor = self.conn.execute('SELECT Time, Altitude, Latitude_Degrees, Longitude_Degrees FROM user_data WHERE Agent_ID = ?', (user_id,))
            return cursor.fetchall()
        
    def get_all_orentation_data_for_user(self, user_id):
        with self.conn:
            cursor = self.conn.execute('SELECT Time,Acceleremoter_X,Acceleremoter_Y,Acceleremoter_Z,Gyroscope_X,Gyroscope_Y,Gyroscope_Z FROM user_data WHERE Agent_ID = ?', (user_id,))
            data = cursor.fetchall()
            return data



