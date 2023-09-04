import pandas as pd
from datetime import datetime

class ContactTracer:
    def __init__(self):
        self.data = pd.DataFrame(columns=["UserID", "Coordinates", "Timestamp"])

    def add_record(self, user_id, coordinates):
        """
        Add a new location record for a user using pandas.concat.
        """
        timestamp = datetime.now()
        new_data = pd.DataFrame([{"UserID": user_id, "Coordinates": coordinates, "Timestamp": timestamp}])
        self.data = pd.concat([self.data, new_data], ignore_index=True)

    def get_users_within_radius(self, target_coordinates, radius):
        """
        Returns the user IDs of users who are within the specified radius of the target coordinates.
        """
        close_users = self.data[self.data["Coordinates"].apply(lambda x: self.distance(x, target_coordinates) <= radius)]
        return close_users["UserID"].unique()

    @staticmethod
    def distance(coord1, coord2):
        """
        Calculate distance between two coordinates. This is a simple Euclidean distance.
        For real-world scenarios, you might want to use geospatial distance calculations.
        """
        return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5

    def get_contacts_of_user(self, user_id, radius):
        """
        Returns a list of unique user IDs who came into contact (within the specified radius) with the given user.
        """
        user_coords = self.data[self.data["UserID"] == user_id]["Coordinates"].tolist()
        contact_ids = []
        for coord in user_coords:
            contact_ids.extend(self.get_users_within_radius(coord, radius))
        # Remove duplicates and the original user ID
        return list(set(contact_ids) - {user_id})
    
    
tracer = ContactTracer()

# Adding some sample records
tracer.add_record("UserA", (1, 2))
tracer.add_record("UserB", (2, 2))
tracer.add_record("UserC", (10, 10))
tracer.add_record("UserA", (3, 2))
tracer.add_record("UserA", (4, 3))
tracer.add_record("UserA", (4, 4))
tracer.add_record("UserD", (4, 5))
tracer.add_record("UserD", (5, 5))
tracer.add_record("UserD", (4, 7))
tracer.add_record("UserD", (4, 8))
tracer.add_record("UserD", (5, 9))
tracer.add_record("UserD", (6, 10))
tracer.add_record("UserD", (8, 11))


# Getting contacts of UserA within a radius of 2 units
contacts = tracer.get_contacts_of_user("UserA", 2)
print(contacts)  # Expected output: ['UserB']
