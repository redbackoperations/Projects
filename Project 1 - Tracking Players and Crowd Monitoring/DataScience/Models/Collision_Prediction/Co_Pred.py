
from math import sqrt, sin, cos, radians
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from math import atan2
import pandas as pd

import sys

sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')

broker_address="test.mosquitto.org"
topic="Orion_test/collision prediction"

from DataManager.MQTTManager import MQTTDataFrameHandler as MQDH 
Handler=MQDH(broker_address, topic)

from Dashboard import Dashboard as DB


class CustomKalmanFilter:
    # [Needs Adjustments for complex features]
    def __init__(self, process_variance, measurement_variance, initial_value=(0, 0), 
                 initial_velocity=(0, 0), initial_estimate_error=1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.velocity = initial_velocity
        self.estimate_error = initial_estimate_error

    def predict(self, acceleration, direction, dt=1.0):
        direction_rad = radians(direction)
        delta_vx = acceleration * cos(direction_rad) * dt
        delta_vy = acceleration * sin(direction_rad) * dt
        self.velocity = (self.velocity[0] + delta_vx, self.velocity[1] + delta_vy)
        self.estimate = (self.estimate[0] + self.velocity[0]*dt + 0.5*delta_vx*dt**2,
                         self.estimate[1] + self.velocity[1]*dt + 0.5*delta_vy*dt**2)
        self.estimate_error += self.process_variance

    def update(self, measurement):
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.estimate = (self.estimate[0] + kalman_gain * (measurement[0] - self.estimate[0]),
                         self.estimate[1] + kalman_gain * (measurement[1] - self.estimate[1]))
        self.estimate_error *= (1 - kalman_gain)

class CollisionPrediction:
    def __init__(self, process_variance, measurement_variance):
        self.users = {}
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        # Store the latest collision predictions
        self.latest_collisions = [] 

    def update_user(self, user_id, coordinates, speed, direction, acceleration):
        vx = speed * cos(radians(direction))
        vy = speed * sin(radians(direction))
        if user_id not in self.users:
            self.users[user_id] = CustomKalmanFilter(self.process_variance, self.measurement_variance, 
                                                     initial_value=coordinates, initial_velocity=(vx, vy))
        else:
            self.users[user_id].predict(acceleration, direction)
            self.users[user_id].update(coordinates)
        
        # Update the latest collision predictions each time user details are updated
        # Using a default prediction_time of 5
        self.latest_collisions = self.predict_collisions(5)  

    def predict_collisions(self, prediction_time, interval=1):
        '''
        Predict collisions at regular intervals within the prediction time.
        '''
        collisions = set()

        #Check for collisions at regular intervals
        for t in range(0, prediction_time + 1, interval):
            user_ids = list(self.users.keys())
            future_positions = {}
            for user_id in user_ids:
                kf = self.users[user_id]
                future_x = kf.estimate[0] + kf.velocity[0]*t
                future_y = kf.estimate[1] + kf.velocity[1]*t
                future_positions[user_id] = (future_x, future_y)
            for i in range(len(user_ids)):
                for j in range(i + 1, len(user_ids)):
                    pos1 = future_positions[user_ids[i]]
                    pos2 = future_positions[user_ids[j]]
                    distance = sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    if distance < 5:
                        collisions.add((user_ids[i], user_ids[j]))
                        
        return list(collisions)



class EnhancedVisualizeMovements:
    # [Needs Adjustments for complex features]
    def __init__(self, collision_prediction):
        self.collision_prediction = collision_prediction

    def generate_random_color(self):
        return (random.random(), random.random(), random.random())
    
   
    def compute_intersection(self,initial_position1, velocity1, initial_position2, velocity2):
        '''
        Compute the intersection point of two trajectories given their initial positions and velocities.
        '''

        # Unpack the initial positions and velocities
        x1, y1 = initial_position1
        vx1, vy1 = velocity1
        x2, y2 = initial_position2
        vx2, vy2 = velocity2

        # Handle the special cases where the trajectories are vertical
        if vx1 == 0 and vx2 == 0:  # Both trajectories are vertical
            return None  # No unique intersection point
        elif vx1 == 0:  # Only the first trajectory is vertical
            x = x1
            y = vy2/vx2 * (x - x2) + y2
            return (x, y)
        elif vx2 == 0:  # Only the second trajectory is vertical
            x = x2
            y = vy1/vx1 * (x - x1) + y1
            return (x, y)

        # Calculate the slopes for the trajectories
        m1 = vy1 / vx1
        m2 = vy2 / vx2

        # If the slopes are equal, the trajectories are parallel and do not intersect
        if m1 == m2:
            return None

        # Compute x-coordinate of intersection
        x = (y2 - y1 + m1*x1 - m2*x2) / (m1 - m2)

        # Compute corresponding y-coordinate using one of the trajectory equations
        y = m1 * (x - x1) + y1

        return (x, y)
    
    def plot_enhanced_movements(self, ax, prediction_time):
        '''
        Visualize user trajectories and potential collisions.
        '''
        
        # Plot initial and predicted positions with intervals
        for user_id, kf in self.collision_prediction.users.items():
            color = user_colors[user_id]  # Use predefined colors for consistency
            
            initial_x, initial_y = kf.estimate
            
            for t in range(0, prediction_time + 1, 1):
                # Recompute direction at each step
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                
            
            
            predicted_x = initial_x + dx * prediction_time
            predicted_y = initial_y + dy * prediction_time
            
            ax.plot(initial_x, initial_y, 's', color=color, markersize=8)
            ax.annotate('Start', (initial_x, initial_y), textcoords="offset points", xytext=(0,5), ha='center')
            
            ax.plot(predicted_x, predicted_y, 'o', color=color, markersize=8)
            ax.annotate('End', (predicted_x, predicted_y), textcoords="offset points", xytext=(0,5), ha='center')
            
            ax.plot([initial_x, predicted_x], [initial_y, predicted_y], color=color, linestyle='-')
            ax.arrow(initial_x, initial_y, predicted_x - initial_x, predicted_y - initial_y, head_width=1, head_length=1, fc=color, ec=color)
        
        collisions = self.collision_prediction.predict_collisions(prediction_time)
        for user1, user2 in collisions:
            future_position1 = self.collision_prediction.users[user1].estimate
            velocity1 = self.collision_prediction.users[user1].velocity
            future_position2 = self.collision_prediction.users[user2].estimate
            velocity2 = self.collision_prediction.users[user2].velocity
                    
            collision_point = self.compute_intersection(future_position1, velocity1, future_position2, velocity2)
            
            if collision_point:  # If there's a unique intersection point
                collision_x, collision_y = collision_point
                ax.plot(collision_x, collision_y, 'ro', markersize=10)               
            
            try:
                data={'user1':user1,'user2':user2,'collision_point':collision_point}
                df=pd.DataFrame(data)
                if df is not None or isinstance(df, pd.DataFrame):
                    Handler.send_data(df)
                    print("Data Sent to Orion:", df)
                        
            except:
                print("Error Sending Data to Orion, current data:",df)
                continue



          
            ax.set_title(f"User Movements: Initial to {prediction_time} Time Units")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True)
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.show()
            ## Send Collision Data to Orion Broker




# Testing the User class with natural movement


NUM_USERS = 10
NUM_ITERATIONS = 100

# Initialize the collision prediction system and visualizer
collision_prediction = CollisionPrediction(0.1,0.1)
visualizer = EnhancedVisualizeMovements(collision_prediction)



# Create the initial plot
fig, ax = plt.subplots()
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_title("User Movements Animation")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

# Initialize user_positions dictionary outside the update function to persist positions across frames
user_positions = {}


# Assigning a fixed color to each user to ensure consistent coloring across frames
user_colors = {user_id: plt.cm.jet(i/NUM_USERS) for i, user_id in enumerate(range(1, NUM_USERS + 1))}

def update(frame):
    ax.clear()
    ax.set_xlim(-1000, 1000)  # Adjusting limits to match the earlier defined space
    ax.set_ylim(-1000, 1000)
    
    for user_id in range(1, NUM_USERS + 1):
        # Check the current position of the user
        current_position = user_positions.get(user_id, (random.randint(0, 100), random.randint(0, 100)))
        
        # Generate incremental movement parameters
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        x = current_position[0] + dx
        y = current_position[1] + dy
        user_positions[user_id] = (x, y)  # Update the new position in the dictionary
        
        speed = (dx**2 + dy**2)**0.5
        direction = (180 / 3.14159) * atan2(dy, dx)
        
        direction = (180 / 3.14159) * atan2(dy, dx)

        # Introduce a random direction shift
        angle_shift = random.uniform(-90, 90)  # Random shift between -30 and 30 degrees
        direction += angle_shift
        acceleration = 0  # Assuming no acceleration for simplicity
        
        # Update user's movement in the collision prediction system
        collision_prediction.update_user(user_id, (x, y), speed, direction, acceleration)
        
        # Plot the user's position with a fixed color
        ax.plot(x, y, 'o', color=user_colors[user_id],label=f"User {user_id}")
   
    # Visualize the movements and potential collisions
    visualizer.plot_enhanced_movements(ax, 10)
    
    


# Create the animation
ani = FuncAnimation(fig, update, frames=range(NUM_ITERATIONS), repeat=False)

plt.show()
