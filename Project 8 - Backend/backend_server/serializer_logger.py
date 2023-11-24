from .models import Users
from .auth_form_serializers import LoginSerializer, SignupSerializer

def print_serialized_data():
    # Fetch data from your database using Django models
    queryset = Users.objects.all()  # Example queryset, modify according to your model

    # Serialize the retrieved data
    serialized_data = auth_form_serializers(queryset, many=True).data

    # Print or log the serialized data
    for data in serialized_data:
        print(data)  # Example: Print serialized data to console
        # Alternatively, you can log it to a file or another logging mechanism

# Execute the function when the file is run directly
if __name__ == "__main__":
    print("Printing Serialized Data:")
    print_serialized_data()
