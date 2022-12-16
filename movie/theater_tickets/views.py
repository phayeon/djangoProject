import json

from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf

from movie.theater_tickets.fashion_service import FashionService
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
    return JsonResponse({'Response Test ': t.hook(req)})


@api_view(['GET', 'POST'])
def fashion(request):
    if request.method == 'GET':
        print(f"######## ID is {request.GET['get_num']} ########")
        return JsonResponse(
            {'result': FashionService().service_model(int(request.GET['get_num']))})
    elif request.method == 'POST':
        data = json.loads(request.body)  # json to dict
        print(f"######## GET at Here ! React ID is {data['post_num']} ########")
        result = FashionService().service_model(int(data['post_num']))
        return JsonResponse({'result': result})
