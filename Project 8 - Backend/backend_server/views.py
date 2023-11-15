from django.http import JsonResponse
from .models import Users
from .serializers import UserSerializer


def user_list(request):
    #get all user
    users = Users.objects.all()
    serilaizer = UserSerializer(users, many=True)
    return JsonResponse(serilaizer.data,  safe=False)

