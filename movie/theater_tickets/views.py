from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf
from movie.theater_tickets.iris_model import IrisModel
from movie.theater_tickets.irls_service import IrisService
from movie.theater_tickets.stroke import StrokeService


@api_view(['GET'])
@parser_classes([JSONParser])
def stroke(request):
    print(f'Stroke Check {request}')
    StrokeService().hook()
    return JsonResponse({'Response Test ': 'SUCCESS'})


@api_view(['GET'])
@parser_classes([JSONParser])
def iris_Get(request):
    IrisModel().spec()
    return JsonResponse({'Response Test ': 'SUCCESS'})


@api_view(['POST'])
@parser_classes([JSONParser])
def iris_Post(request):
    iris_info = request.data
    sl = tf.constant(float(iris_info['sl']))
    sw = tf.constant(float(iris_info['sw']))
    pl = tf.constant(float(iris_info['pl']))
    pw = tf.constant(float(iris_info['pw']))
    req = [sl, sw, pl, pw]
    t = IrisService()
    print(f'리액트에서 받아 온 데이터 : {request}')
    print(f'꽃받침 길이 : {sl}')
    print(f'꽃받침 넓이 : {sw}')
    print(f'꽃잎 길이 : {pl}')
    print(f'꽃잎 넓이 : {pw}')
    t.hook(req)
    return JsonResponse({'Response Test ': 'SUCCESS'})
# Create your views here.
