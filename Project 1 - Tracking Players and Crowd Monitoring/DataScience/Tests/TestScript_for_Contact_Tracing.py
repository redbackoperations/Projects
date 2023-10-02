#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
from datetime import datetime, timedelta
import pandas as pd
from Contact_Tracing.py import ContactTracer  # Replace 'your_script_name' with the actual filename of your script

class TestContactTracer(unittest.TestCase):

    def setUp(self):
        # Create a ContactTracer instance for testing
        self.tracer = ContactTracer()

    def test_add_record(self):
        # Test the add_record method
        timestamp = datetime.now()
        self.tracer.add_record("UserA", (1, 2), timestamp)
        self.assertEqual(len(self.tracer.data), 1)

    def test_get_time_based_contacts(self):
        # Test the get_time_based_contacts method
        
        # Populate the tracer with sample data
        records = [
            ("UserA", (1, 2)),
            ("UserB", (2, 2)),
            ("UserC", (10, 10)),
            ("UserA", (3, 2)),
            # Add more sample records as needed
        ]
        base_timestamp = datetime.now()
        for i, (user, coords) in enumerate(records):
            timestamp = base_timestamp + timedelta(minutes=i)
            self.tracer.add_record(user, coords, timestamp)
        
        # Test contacts for UserA within a radius of 2 units and a time window of 30 minutes
        contacts = self.tracer.get_time_based_contacts("UserA", 2)
        self.assertEqual(len(contacts), expected_contact_count)  # Replace expected_contact_count with the expected value

if __name__ == '__main__':
    unittest.main()


# In[ ]:




