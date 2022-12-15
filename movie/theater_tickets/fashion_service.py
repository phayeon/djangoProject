import os.path

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow import keras

'''
Iris Species
Classify iris plants into three species in this classic dataset
'''


class FashionService:
    def __init__(self):
        global class_names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def hook(self, feature):
        self.service_model(feature)

    def service_model(self, feature):
        model = load_model(os.path.join(os.path.abspath("save"), "fashion_model.h5"))
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        predictions = model.predict(test_images)
        predictions_array, true_label, img = predictions[feature], test_labels[feature], test_images[feature]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array)
        print(f'예측한 답 : {predicted_label}')
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel('{} {:2.0f}% ({})'.format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label]
        ), color=color)
        # plt.show()


FASHION_MENUS = ["종료", "보기"]
fashion_menu = {
    "1": lambda x: x.hook()
}

if __name__ == '__main__':
    def my_menu(ls):
        for i, j in enumerate(ls):
            print(f"{i}. {j}")
        return input('메뉴 선택: ')

    t = FashionService()
    while True:
        menu = my_menu(FASHION_MENUS)
        if menu == '0':
            print("종료")
            break
        else:
            fashion_menu[menu](t)
