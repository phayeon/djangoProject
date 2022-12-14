import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

'''
Iris Species
Classify iris plants into three species in this classic dataset
'''


class IrisModel:
    def __init__(self):
        self.iris = datasets.load_iris()
        print(f'type {type(self.iris)}')  # type <class 'sklearn.utils._bunch.Bunch'>
        self._X = self.iris.data
        self._Y = self.iris.target

    def hook(self):
        self.spec()
        # self.creat_model()

    def spec(self):
        print(" --- 1.Features ---")
        print(self.iris['feature_names'])
        print(" --- 2.target ---")
        print(self.iris['target'])
        print(" --- 3.print ---")
        print(self.iris)

    '''
    --- 1.Shape ---
    (150, 6)
     --- 2.Features ---
    Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    '''

    def creat_model(self):
        X = self._X
        Y = self._Y
        enc = OneHotEncoder()
        Y_1hot = enc.fit_transform(Y.reshape(-1, 1)).toarray()
        model = Sequential()
        model.add(Dense(4, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, Y_1hot, epochs=300, batch_size=10)
        print('Model Training is completed')

        file_name = r'C:\Users\AIA\PycharmProjects\djangoProject\movie\theater_tickets\save\iris_model.h5'
        model.save(file_name)
        print(f'Model Saved in {file_name}')



IRIS_MENUS = ["종료", "보기"]
iris_menu = {
    "1": lambda x: x.hook()
}

if __name__ == '__main__':
    def my_menu(ls):
        for i, j in enumerate(ls):
            print(f"{i}. {j}")
        return input('메뉴 선택: ')

    t = IrisModel()
    while True:
        menu = my_menu(IRIS_MENUS)
        if menu == '0':
            print("종료")
            break
        else:
            iris_menu[menu](t)
