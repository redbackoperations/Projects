from rest_framework import serializers
from .models import Users

# Serializer for the Users model to convert Python objects to JSON
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Users  # Specifies the model to be serialized
        fields = ['id', 'username', 'password']  # Specifies the fields to include in the serialized representation


