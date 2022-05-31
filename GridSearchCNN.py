import tensorflow as tf


class GridSearchCNN:
    @staticmethod
    def create_model(dropout_rate, num_class):
        # create model
        # feel free to change this and fit it according to your model
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                   input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # add a dropout layer if rate is not null
        if dropout_rate != 0:
            model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
        # add a dropout layer if rate is not null
        if dropout_rate != 0:
            model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model.add(tf.keras.layers.Dense(num_class, activation='softmax'))

        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    # define function to display the results of the grid search
    @staticmethod
    def display_cv_results(search_results):
        print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
        means = search_results.cv_results_['mean_test_score']
        stds = search_results.cv_results_['std_test_score']
        params = search_results.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))
