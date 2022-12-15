import os

import keras.datasets.fashion_mnist
import matplotlib.pyplot as plt
from keras import Sequential


class FashionModel(object):


    def hook(self):
        self.creat_model()

    def creat_model(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        plt.figure()
        plt.imshow(train_images[3])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        model = Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=5)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Test Accuracy is {test_acc}')
        file_name = os.path.join(os.path.abspath("save"), "fashion_model.h5")
        print(f'Model Save {file_name}')
        model.save(file_name)
        return model


FASHION_MENUS = ["종료", "보기"]
fashion_menu = {
    "1": lambda x: x.hook()
}

if __name__ == '__main__':
    def my_menu(ls):
        for i, j in enumerate(ls):
            print(f"{i}. {j}")
        return input('메뉴 선택: ')

    t = FashionModel()
    while True:
        menu = my_menu(FASHION_MENUS)
        if menu == '0':
            print("종료")
            break
        else:
            fashion_menu[menu](t)
