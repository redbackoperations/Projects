
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import seaborn as sns
import folium
from folium.plugins import HeatMap

import sys

sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')

broker_address="test.mosquitto.org"
topic="Orion_test/Overcrowding Detection"

from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH 
Handler=MQDH(broker_address, topic)

from Dashboard import Dashboard as DB

def process_data(gps_data):
    """
    Processes the GPS data using DBSCAN clustering and plots the clusters.

    :param gps_data: A numpy array with GPS coordinates in the format [[latitude1, longitude1], [latitude2, longitude2], ...].
    """
    # Using DBSCAN to cluster the data
    dbscan = DBSCAN(eps=0.01, min_samples=5)
    clusters = dbscan.fit_predict(gps_data)
    
    df = pd.DataFrame({'Latitude': gps_data[:, 0], 'Longitude': gps_data[:, 1], 'Cluster': clusters})
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    plt.figure(figsize=(10, 10))
    for cluster_id, color in zip(set(clusters), colors):
        if cluster_id == -1:
            continue
        cluster_points = df[df['Cluster'] == cluster_id]
        plt.scatter(cluster_points['Longitude'], cluster_points['Latitude'], c=[color], label=f'Cluster {cluster_id}')

    plt.title('Clusters of Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()
    return  df

# Initialize empty arrays for latitudes and longitudes
latitudes = []
longitudes = []

# Define cluster centers
cluster_centers = [ 
    (51.507351, -0.127758),   # London, UK   
]

# Define standard deviation for the distribution of points around the cluster center
std_dev = 0.03

# Number of points per cluster
points_per_cluster = 100

# Generate points for cluster
for center in cluster_centers:
    lat_center, lon_center = center
    latitudes += list(np.random.normal(lat_center, std_dev, points_per_cluster))
    longitudes += list(np.random.normal(lon_center, std_dev, points_per_cluster))

# reading latitudes and longitudes from the VirtualCrowd_Test_Cleaned.csv 
#change this line depending on the file path
df = pd.read_csv(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Clean Datasets\VD2.csv')

time = df[df["Time"].isin(['9:25:47'])].reset_index(drop=True)
latitudes_list = df[" Longitude Degrees"].tolist()
longitudes_list = df[" Latitude Degrees"].tolist()

# remove "Time" to facilitate the processing
df.drop("Time", axis=1)

latitudes = latitudes_list
longitudes = longitudes_list

# Processing the data and plotting
gps_data = np.array([latitudes, longitudes]).T
cluster_data=process_data(gps_data)

# Creating a DataFrame with the latitude and longitude data
heatmap_data = pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes})

# Plotting the heatmap using Seaborn's kdeplot function
plt.figure(figsize=(10, 10))
sns.kdeplot(x='Longitude', y='Latitude', data=heatmap_data, cmap='Reds', fill=True)
plt.title('Heatmap of Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Calculate the mean latitude and longitude
mean_latitude = np.mean(latitudes)
mean_longitude = np.mean(longitudes)

# Create a base map, centered at the mean coordinates
base_map = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=13)

# Create a list of [lat, lon] pairs for the heatmap
heatmap_data = [[lat, lon] for lat, lon in zip(latitudes, longitudes)]

# Add the heatmap layer to the base map
HeatMap(heatmap_data).add_to(base_map)

# Save the map to an HTML file (optional)
#change this line depending on the file path
base_map.save(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Overcrowding Detection\heatmap.html')

Handler.send_data(cluster_data)



