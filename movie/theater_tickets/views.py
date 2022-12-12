from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
from movie.theater_tickets.stroke import StrokeService


@api_view(['GET'])
@parser_classes([JSONParser])
def stroke(request):
    print(f'Stroke Check {request}')
    StrokeService().hook()
    return JsonResponse({'Response Test ': 'SUCCESS'})
# Create your views here.
