import pandas as pd
from faker import Faker
from hashlib import sha256

# Read the original dataset
data = pd.read_csv(r"C:\Users\GondesisivaramSantos\Downloads\heart_attack_prediction_dataset.csv")

# Initialize Faker instance for generating fake data
fake = Faker()

# Anonymize or mask each column
for column in data.columns:
    if column == 'Patient ID':
        # Hash the patient ID column using SHA-256
        data[column] = [sha256(str(fake.unique.random_number()).encode('utf-8')).hexdigest() for _ in range(len(data))]
    elif column == 'Age':
        # Generalize age ranges
        data[column] = data[column].apply(lambda x: x // 10 * 10)
    elif column in ['Sex', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption',
                    'Previous Heart Problems', 'Medication Use']:
        # Mask binary attributes (1s and 0s)
        data[column] = data[column].apply(lambda x: 'Yes' if x == 1 else 'No')
    elif column == 'Heart Attack Risk':
        # Mask heart attack risk (0 and 1)
        data[column] = data[column].apply(lambda x: 'High' if x == 1 else 'Low')
    elif column in ['Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours Per Week',
                    'Stress Level', 'Income', 'BMI', 'Triglycerides']:
        # Replace numeric values with random values within a range
        data[column] = data[column].apply(lambda x: fake.random_int(min=1, max=1000))
    else:
        # Mask non-sensitive columns by generating fake data
        data[column] = [fake.pystr() for _ in range(len(data))]

# Save the anonymized dataset
data.to_csv('anonymized_dataset.csv', index=False)