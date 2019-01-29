import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

print('TF version: ', tf.__version__)

        ############################################################################################
        #                                        base classification                               #
        # ##########################################################################################


def base_classification():
    # downoload Fashion MNIST
    fashion_mnist = keras.datasets.fashion_mnist
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    # scale data to 0 - 1 range
    train_x = train_x / 255
    test_x = test_x / 255

    # display images to see if it's OK
    # display_images(train_x, train_y, class_names)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # just transforms 2 dimentional array to one dimentional 28*28=784
        keras.layers.Dense(128, activation=tf.nn.relu),  # normal dense layer with 128 neurons
        keras.layers.Dense(10, activation=tf.nn.softmax)  # output layer with softmax activation function and 10
        # (classes score) neurons output
    ])

    # compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # train model
    model.fit(train_x, train_y, epochs=10)

    # test model on test data
    test_loss, tes_accuracy = model.evaluate(test_x, test_y)
    print('Test Accuracy is:', tes_accuracy)

    # predict model on single image
    predict_class(model, class_names, test_x[0])


def display_images(x_train, y_train, classes):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(classes[y_train[i]])

    plt.show()


def predict_class(model, classes, x_test):
    # make X same shape as training examples (batches)
    x_test = (np.expand_dims(x_test, 0))


    predictions = model.predict(x_test)

    for pred in predictions:
        score = np.argmax(pred)
        print(f'Score: {score}, Class is: {classes[score]}')



        ############################################################################################
        #                                        text classification                               #
        # ##########################################################################################




base_classification()
