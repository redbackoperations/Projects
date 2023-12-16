from rest_framework import serializers

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True)

class SignupSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password1 = serializers.CharField(write_only=True)
    password2 = serializers.CharField(write_only=True)

    def validate(self, data):
        # Custom validation logic for the signup form fields
        print(data['password1'])
        print(data['password2'])
        if data['password1'] != data['password2']:
            raise serializers.ValidationError("Passwords do not match")
        return data