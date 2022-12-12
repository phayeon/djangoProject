from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from movie.movies.services import DcGan

@api_view(['GET'])
@parser_classes([JSONParser])
def faces(request):
    print(f'Enter Show Faces with {request}')
    return JsonResponse({'Response Test ': 'SUCCESS'})
# Create your views here.
