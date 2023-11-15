from rest_framework import serializers
from .models import Users

#Going from python to json
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Users
        fields= ['id','username', 'password']

