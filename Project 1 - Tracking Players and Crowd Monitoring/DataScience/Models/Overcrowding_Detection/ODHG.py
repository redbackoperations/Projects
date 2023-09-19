
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import seaborn as sns
import folium
from folium.plugins import HeatMap
import panel as pn
import sys

sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')

broker_address="test.mosquitto.org"
topic="Orion_test/Overcrowding Detection"

from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH 
Handler=MQDH(broker_address, topic)



def process_data(gps_data):
    """
    Processes the GPS data using DBSCAN clustering and plots the clusters.

    :param gps_data: A numpy array with GPS coordinates in the format [[latitude1, longitude1], [latitude2, longitude2], ...].
    """
    # Using DBSCAN to cluster the data
    dbscan = DBSCAN(eps=0.02, min_samples=15, metric='euclidean')
    clusters = dbscan.fit_predict(gps_data)
    print(len(clusters))
    
    df = pd.DataFrame({'Latitude': gps_data[:, 0], 'Longitude': gps_data[:, 1], 'Cluster': clusters})
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    figure=plt.figure(figsize=(10, 10))
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
    return  df,figure

# Initialize empty arrays for latitudes and longitudes
latitudes = []
longitudes = []




# reading latitudes and longitudes from the VirtualCrowd_Test_Cleaned.csv 
#change this line depending on the file path
df = pd.read_csv(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Clean Datasets\VD3.csv')

selected_time_data = df[df["Time"] == '22:22:01']
time = df[df["Time"].isin(['22:22:01'])].reset_index(drop=True)
latitudes_list = time[" Longitude Degrees"].tolist()
longitudes_list = time[" Latitude Degrees"].tolist()

# remove "Time" to facilitate the processing
df.drop("Time", axis=1)

latitudes = latitudes_list
longitudes = longitudes_list

# Processing the data and plotting
gps_data = np.array([latitudes, longitudes]).T
cluster_data,fig1=process_data(gps_data)

# Creating a DataFrame with the latitude and longitude data
heatmap_data = pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes})

# Plotting the heatmap using Seaborn's kdeplot function
figure2=plt.figure(figsize=(10, 10))
sns.kdeplot(x='Longitude', y='Latitude', data=heatmap_data, cmap='Reds', fill=True)
plt.title('Heatmap of Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Calculate the mean latitude and longitude for centering the map
mean_latitude = selected_time_data[" Latitude Degrees"].mean()
mean_longitude = selected_time_data[" Longitude Degrees"].mean()

# Create a base map, centered at the mean coordinates
base_map = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=13)

# Create a list of [lat, lon] pairs for the heatmap
heatmap_data = [[lat, lon] for lat, lon in zip(selected_time_data[" Latitude Degrees"], selected_time_data[" Longitude Degrees"])]

# Add the heatmap layer to the base map with adjusted parameters
HeatMap(heatmap_data, radius=15, max_zoom=13, blur=15).add_to(base_map)

# Save the map to an HTML file (optional)
# change this line depending on the file path
base_map.save(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Overcrowding_Detection\heatmap.html')

# Handler.send_data(cluster_data)#send the data to the broker

# plt.close(fig1)
# plt.close(figure2)
plot_panel = pn.pane.Matplotlib(fig1, tight=True)
plot_panel2 = pn.pane.Matplotlib(figure2, tight=True)

