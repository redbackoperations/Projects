from django.db import models 
#user model that takes 2 inputs
class Users(models.Model):
    username= models.CharField(max_length=20)
    password = models.CharField(max_length=20)