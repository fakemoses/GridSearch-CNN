import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from GridSearchCNN import GridSearchCNN


# load train and test dataset
# here testing the method with MNIST dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY


gs = GridSearchCNN()

x_train, y_train, x_test, y_test = load_dataset()

# normalizing inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

num_classes = y_test.shape[1]

n_epochs = 30
n_epochs_cv = 10
n_cv = 3

model = KerasClassifier(model=gs.create_model, verbose=1, dropout_rate=0.0, num_class=num_classes)

# define parameters that needs to be tested using GridSearch Algo
# for this CNN Problem, 3 params were identified and listed as an array

param_grid = {
    'dropout_rate': [0.0, 0.10, 0.20, 0.30],
    'batch_size': [16, 32, 64],
    'epochs': [n_epochs_cv]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)
grid_result = grid.fit(x_train, y_train)  # fit the full dataset as we are using cross validation

# print results
gs.display_cv_results(grid_result)

#you can also retrain the model using the best params from the grid_result and save it into a file
