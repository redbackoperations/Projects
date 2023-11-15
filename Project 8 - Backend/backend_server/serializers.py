from rest_framework import serializers
from .models import Users

#Going from python to json. This class takes the previously made credintial and turn it to Json
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Users
        fields= ['id','username', 'password']

