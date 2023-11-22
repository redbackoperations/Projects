from django.http import JsonResponse
from .models import Users
from .serializers import UserSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status 


@api_view(['GET', 'POST'])

def user_list(request):
    if request.method == 'GET':
         #get all user
        users = Users.objects.all()
        #Serialaize all users
        serilaizer = UserSerializer(users, many=True)
        return JsonResponse(serilaizer.data,  safe=False)

    if request.method == 'POST':
       serializer = UserSerializer(data=request.data)
       if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)