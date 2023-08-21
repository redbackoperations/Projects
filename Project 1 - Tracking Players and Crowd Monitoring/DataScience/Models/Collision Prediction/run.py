from math import sqrt, sin, cos, radians

class CustomKalmanFilter:
    """
    Implementation of a custom Kalman Filter for estimating object's position and velocity.
    """
    
    def __init__(self, process_variance, measurement_variance, initial_value=(0, 0), initial_velocity=(0, 0), initial_estimate_error=1):
        """
        Initialize the Kalman filter with given parameters.

        :param process_variance: float, variance in the process
        :param measurement_variance: float, variance in the measurements
        :param initial_value: tuple, initial position (x, y)
        :param initial_velocity: tuple, initial velocity (vx, vy)
        :param initial_estimate_error: float, initial estimate error
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.velocity = initial_velocity
        self.estimate_error = initial_estimate_error

    def predict(self, acceleration, direction, dt=1.0):
        """
        Prediction step of the Kalman filter.

        :param acceleration: float, acceleration
        :param direction: float, direction in degrees
        :param dt: float, time step (default is 1.0)
        """
        direction_rad = radians(direction)
        # Calculate changes in velocity components
        delta_vx = acceleration * cos(direction_rad) * dt
        delta_vy = acceleration * sin(direction_rad) * dt
        # Update velocity and position estimates
        self.velocity = (self.velocity[0] + delta_vx, self.velocity[1] + delta_vy)
        self.estimate = (self.estimate[0] + self.velocity[0]*dt + 0.5*delta_vx*dt**2,
                         self.estimate[1] + self.velocity[1]*dt + 0.5*delta_vy*dt**2)
        # Update estimate error
        self.estimate_error += self.process_variance

    def update(self, measurement):
        """
        Update step of the Kalman filter.

        :param measurement: tuple, measured position (x, y)
        """
        # Calculate Kalman gain
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        # Update estimate and error
        self.estimate = (self.estimate[0] + kalman_gain * (measurement[0] - self.estimate[0]),
                         self.estimate[1] + kalman_gain * (measurement[1] - self.estimate[1]))
        self.estimate_error *= (1 - kalman_gain)

class CollisionPrediction:
    """
    Collision prediction system using custom Kalman Filters for tracking users.
    """
    
    def __init__(self, process_variance, measurement_variance):
        """
        Initialize the system with given parameters.

        :param process_variance: float, variance in the process
        :param measurement_variance: float, variance in the measurements
        """
        self.users = {}
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def update_user(self, user_id, coordinates, speed, direction, acceleration):
        """
        Update user's position, velocity, and acceleration.

        :param user_id: int, unique identifier for the user
        :param coordinates: tuple, user's coordinates (x, y)
        :param speed: float, user's speed
        :param direction: float, user's direction in degrees
        :param acceleration: float, user's acceleration
        """
        # Calculate velocity components
        vx = speed * cos(radians(direction))
        vy = speed * sin(radians(direction))
        # Create or update Kalman Filter for user
        if user_id not in self.users:
            self.users[user_id] = CustomKalmanFilter(self.process_variance, self.measurement_variance, 
                                                     initial_value=coordinates, initial_velocity=(vx, vy))
        else:
            self.users[user_id].predict(acceleration, direction)
            self.users[user_id].update(coordinates)

    def predict_collisions(self, prediction_time):
        """
        Predict collisions among users in given prediction time.

        :param prediction_time: float, time in future to predict
        :return: list, pairs of user_ids predicted to collide
        """
        collisions = []
        user_ids = list(self.users.keys())
        future_positions = {}
        # Predict future positions
        for user_id in user_ids:
            kf = self.users[user_id]
            future_x = kf.estimate[0] + kf.velocity[0]*prediction_time
            future_y = kf.estimate[1] + kf.velocity[1]*prediction_time
            future_positions[user_id] = (future_x, future_y)

        # Check for collisions
        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                pos1 = future_positions[user_ids[i]]
                pos2 = future_positions[user_ids[j]]
                distance = sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                # If distance is less than 5 units, append to collisions
                if distance < 5:
                    collisions.append((user_ids[i], user_ids[j]))

        return collisions
