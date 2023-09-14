import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')
broker_address="test.mosquitto.org"
topic="Orion_test/contact tracing"


from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH 
Handler=MQDH(broker_address, topic)

class ContactTracer:
    def __init__(self):
        self.data = pd.DataFrame(columns=["UserID", "Coordinates", "Timestamp"])

    def add_record(self, user_id, coordinates, timestamp=None):
        """Add a new location record for a user using pandas.concat."""
        if timestamp is None:
            timestamp = datetime.now()
        new_record = pd.DataFrame({"UserID": [user_id], "Coordinates": [coordinates], "Timestamp": [timestamp]})
        self.data = pd.concat([self.data, new_record], ignore_index=True)
        
    def get_time_based_contacts(self, user_id, radius, time_window=timedelta(minutes=30)):
        """Get users that have been in contact with the specified user based on time, without repetition."""
        user_data = self.data[self.data["UserID"] == user_id]
        potential_contacts = pd.DataFrame()

        for _, record in user_data.iterrows():
            lat1, lon1 = record["Coordinates"]
            timestamp = record["Timestamp"]
            contacts = self.data[
                (self.data["Timestamp"] - timestamp).abs() <= time_window  # time condition
            ]
            for _, contact in contacts.iterrows():
                lat2, lon2 = contact["Coordinates"]
                distance = ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5  # simple distance formula
                if distance <= radius and contact["UserID"] != user_id:
                    potential_contacts = pd.concat([potential_contacts, contact.to_frame().T], ignore_index=True)

        # Drop duplicate rows
        potential_contacts = potential_contacts.drop_duplicates()

        return potential_contacts

    

# Re-populate the ContactTracer instance with the previously generated data (with timestamps)
# Repopulating the ContactTracer instance with the provided data and adding timestamps

# Resetting the ContactTracer instance
tracer = ContactTracer()

# Adding records with timestamps
records = [
    ("UserA", (1, 2)),
    ("UserB", (2, 2)),
    ("UserC", (10, 10)),
    ("UserA", (3, 2)),
    ("UserA", (4, 3)),
    ("UserA", (4, 4)),
    ("UserD", (4, 5)),
    ("UserD", (5, 5)),
    ("UserD", (4, 7)),
    ("UserD", (4, 8)),
    ("UserD", (5, 9)),
    ("UserD", (6, 10)),
    ("UserD", (8, 11)),
    ("UserE", (9, 12)),
    ("UserE", (10, 12))
]

# Assigning a unique timestamp for each record
base_timestamp = datetime.now()
for i, (user, coords) in enumerate(records):
    timestamp = base_timestamp + timedelta(minutes=i)  # Each record is 1 minute apart
    tracer.add_record(user, coords, timestamp)


#getting all user contacts to send to the Orion broker
for user in tracer.data["UserID"].unique():
    contacts = tracer.get_time_based_contacts(user, 2)
    print(f"Contacts of {user}:")
    print(contacts)
    df=pd.DataFrame(contacts)
    Handler.send_data(df, user_id=user)


# Getting contacts of UserA within a radius of 2 units
contacts = tracer.get_time_based_contacts("UserA", 2)



def plot_spatial_temporal_contacts(central_user, contacts_df):
    plt.figure(figsize=(12, 10))
    
    # Plot the central user at (0,0) for simplicity
    plt.scatter(0, 0, color="red", label=central_user, s=200, zorder=5)
    plt.text(0, 0, central_user, fontsize=12, ha='right')
    
    # Plot the contacts
    colors = plt.cm.rainbow(np.linspace(0, 1, len(contacts_df["UserID"].unique())))
    color_map = dict(zip(contacts_df["UserID"].unique(), colors))
    
    for _, row in contacts_df.iterrows():
        x, y = row["Coordinates"]
        user = row["UserID"]
        plt.scatter(x, y, color=color_map[user], s=100)
        plt.text(x, y, user, fontsize=12, ha='right')
        
        # Draw a line between the central user and the contact
        plt.plot([0, x], [0, y], color=color_map[user], linestyle='--', linewidth=1)
        
        # Annotate the line with the timestamp of contact
        midpoint = ((x+0)/2, (y+0)/2)
        plt.annotate(row["Timestamp"].strftime('%H:%M:%S'), 
                     xy=midpoint, 
                     xytext=midpoint, 
                     fontsize=10,
                     arrowprops=dict(facecolor='black', arrowstyle='-'),
                     ha='center')
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Spatial-Temporal Contacts of {central_user}")
    plt.grid(True)
    plt.show()

# Plotting the contacts of "UserA"
plot_spatial_temporal_contacts("UserA", contacts)
