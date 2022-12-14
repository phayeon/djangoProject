import numpy as np
from keras.models import load_model
import tensorflow as tf
from sklearn import datasets

'''
Iris Species
Classify iris plants into three species in this classic dataset
'''


class IrisService:
    def __init__(self):
        global model, graph, target_names
        model = load_model(r'C:\Users\AIA\PycharmProjects\djangoProject\movie\theater_tickets\save\iris_model.h5')
        target_names = datasets.load_iris().target_names

    def hook(self, features):
        self.service_model(features)

    def service_model(self, features):
        features = np.reshape(features, (1, 4))
        Y_pred = model.predict(features, verbose=0)
        predicted = Y_pred.argmax(axis=-1)
        if predicted == 0:
            print('setosa / 부채붓꽃')
        elif predicted == 1:
            print('versicolor / 버시칼라 ')
        elif predicted == 2:
            print('virginica / 버지니카')
        return predicted[0]


IRIS_MENUS = ["종료", "보기"]
iris_menu = {
    "1": lambda x: x.hook()
}

if __name__ == '__main__':
    def my_menu(ls):
        for i, j in enumerate(ls):
            print(f"{i}. {j}")
        return input('메뉴 선택: ')

    t = IrisService()
    while True:
        menu = my_menu(IRIS_MENUS)
        if menu == '0':
            print("종료")
            break
        else:
            iris_menu[menu](t)
