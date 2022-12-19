from django.urls import re_path as url
from movie.theater_tickets import views

urlpatterns = [
    url(r'stroke', views.stroke),
    url(r'irispost', views.iris_Post),
    url(r'irisget', views.iris_Get),
    url(r'fashion', views.fashion),
    url(r'number', views.number)
]

"""

url(r'get/fashionget/(?P<id>)$', views.fashion_Get)
"""