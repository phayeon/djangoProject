import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt



class NumberModel(object):
    def __init__(self):
        pass

    def creat_model(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1)
        test_loss, test_acc = model.evaluate(x_test, y_test)

        model.save(r"C:\Users\AIA\PycharmProjects\djangoProject\movie\theater_tickets\save\number_model.h5")
        return test_acc


class NumberService:
    def __init__(self):
        pass

    def hook(self):
        self.service_model()

    def service_model(self) -> int:
        model = load_model(r'C:\Users\AIA\PycharmProjects\djangoProject\movie\theater_tickets\save\number_model.h5')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        predictions = model.predict(x_test)
        plt.figure(figsize=(6, 3))

        '''
        image
        '''
        plt.subplot(1, 2, 1)
        plt.imshow(x_test[0], cmap=plt.cm.binary)

        '''
        histogram
        '''
        plt.subplot(1, 2, 2)
        plt.xticks(range(10))
        plt.yticks(np.arange(0, 1.1, 0.1))
        thisplot = plt.bar(range(10), predictions[0], color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions[0])
        thisplot[predicted_label].set_color('red')

        plt.show()


MENUS = ["종료", "보기"]
menulist = {
    "1": lambda x: x.hook()
}

if __name__ == '__main__':
    def my_menu(ls):
        [print(f"{i}. {j}") for i, j in enumerate(ls)]
        return input('메뉴 선택: ')

    t = NumberService()
    while True:
        menu = my_menu(MENUS)
        if menu == '0':
            print("종료")
            break
        else:
            menulist[menu](t)
