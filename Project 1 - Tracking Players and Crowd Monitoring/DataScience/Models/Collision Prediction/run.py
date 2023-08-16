from math import sqrt
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0, initial_estimate_error=1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = initial_estimate_error

    def predict(self):
        # Prediction Step
        # In the basic 1D Kalman filter, the prediction is just the last estimate
        self.estimate_error += self.process_variance

    def update(self, measurement):
        # Update Step
        # Calculate the Kalman gain
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        
        # Update the estimate and the estimate error
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.estimate_error *= (1 - kalman_gain)



class CollisionPrediction:
    def __init__(self, process_variance, measurement_variance):
        self.users = {}
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def update_user(self, user_id, coordinates, speed, direction):
        # Use a Kalman Filter to track the state of each user
        if user_id not in self.users:
            self.users[user_id] = KalmanFilter(self.process_variance, self.measurement_variance, initial_value=coordinates)
        else:
            self.users[user_id].predict()
            self.users[user_id].update(coordinates)

    def predict_collisions(self, threshold_distance):
        collisions = []
        user_ids = list(self.users.keys())

        # Compare every pair of users to see if they are on a collision course
        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                user1 = self.users[user_ids[i]]
                user2 = self.users[user_ids[j]]

                distance = sqrt((user1.estimate[0] - user2.estimate[0])**2 + (user1.estimate[1] - user2.estimate[1])**2)

                if distance < threshold_distance:
                    collisions.append((user_ids[i], user_ids[j]))

        return collisions
