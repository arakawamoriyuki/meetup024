# 3DCNN 3次元畳み込み

def get_model(trainer):
    """Returns the model (not compiled, not trained)

    Parameters
    ----------
    trainer : dict
        trainer.set_status: Method allowing you to set the current job status
        logger: Logger with 'write' method allowing you to write into the log
        settings: dict - job settings
        job: dict - job with all information
    """
    __author__ = 'Minhaz Palasara'

    from keras.datasets import shapes_3d
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution3D, MaxPooling3D
    from keras.optimizers import SGD, RMSprop
    from keras.utils import np_utils, generic_utils



    """
        To classify/track 3D shapes, such as human hands (http://www.dbs.ifi.lmu.de/~yu_k/icml2010_3dcnn.pdf),
        we first need to find a distinct set of features. Specifically for 3D shapes, robust classification can be done using
        3D features.
        Features can be extracted by applying a 3D filters. We can auto learn these filters using 3D deep learning.
        This example trains a simple network for classifying 3D shapes (Spheres, and Cubes).
        GPU run command:
            THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python shapes_3d_cnn.py
        CPU run command:
            THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python shapes_3d_cnn.py
        For 4000 training samples and 1000 test samples.
        90% accuracy reached after 40 epochs, 37 seconds/epoch on GTX Titan
    """

    # number of convolutional filters to use at each layer
    nb_filters = [16, 32]

    # level of pooling to perform at each layer (POOL x POOL)
    nb_pool = [3, 3]

    # level of convolution to perform at each layer (CONV x CONV)
    nb_conv = [7, 3]

    c1 = keras.layers.convolutional.Convolution3D(nb_filters[0],nb_depth=nb_conv[0],
            nb_row=nb_conv[0], nb_col=nb_conv[0], border_mode='full',
            input_shape=(1, patch_size, patch_size, patch_size), activation='relu')(input)
    c2 = keras.layers.convolutional.MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0]))(c1)
    c3 = keras.layers.core.Dropout(0.5)(c2)
    c4 = keras.layers.convolutional.Convolution3D(nb_filters[1],nb_depth=nb_conv[1],
            nb_row=nb_conv[1], nb_col=nb_conv[1], border_mode='full', activation='relu')(c3)
    c5 = keras.layers.convolutional.MaxPooling3D(pool_size=(nb_pool[1], nb_pool[1], nb_pool[1]))(c4)
    c6 = keras.layers.core.Flatten()(c5)
    c7 = keras.layers.core.Dropout(0.5)(c6)
    c8 = keras.layers.core.Dense(16, init='normal', activation='relu')(c7)
    c9 = keras.layers.core.Dense(nb_classes, init='normal')(c8)
    c10 = keras.layers.core.Activation('softmax')(c9)

    return keras.models.Model([input], [c10])




def compile(trainer, model, loss, optimizer):
    """Compiles the given model (from get_model) with given loss (from get_loss) and optimizer (from get_optimizer)

    Parameters
    ----------
    trainer : dict
        trainer.set_status: Method allowing you to set the current job status
        logger: Logger with 'write' method allowing you to write into the log
        settings: dict - job settings
        job: dict - job with all information
    """

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])



def get_training_data(trainer, dataset):
    """Returns the training and validation data from dataset. Since dataset is living in its own domain
    it could be necessary to transform the given dataset to match the model's input layer. (e.g. convolution vs dense)

    Parameters
    ----------
    trainer : dict
        trainer.set_status: Method allowing you to set the current job status
        logger: Logger with 'write' method allowing you to write into the log
        settings: dict - job settings
        job: dict - job with all information

    dataset: dict
        Contains for each used dataset a key => dictionary, where key is the dataset id and the dict is returned from the Dataset "get_data" method.
        Their keys: 'X_train', 'Y_train', 'X_test', 'Y_test'.
        Example data['mnistExample']['X_train'] for a input layer that has 'mnistExample' as chosen dataset.
    """

    return {
        'x': {'Input': dataset['AETROS/dataset/VGG-16']['X_train']},
        'y': {'dense_3': dataset['AETROS/dataset/VGG-16']['Y_train']}
    }


def get_validation_data(trainer, dataset):
    """Returns the training and validation data from dataset.

    Parameters
    ----------
    trainer : dict
        trainer.set_status: Method allowing you to set the current job status
        logger: Logger with 'write' method allowing you to write into the log
        settings: dict - job settings
        job: dict - job with all information

    dataset: dict
        Contains for each used dataset a key => dictionary, where key is the dataset id and the dict is returned from the Dataset "get_data" method.
        Their keys: 'X_train', 'Y_train', 'X_test', 'Y_test'.
        Example data['mnistExample']['X_test'] for a input layer that has 'mnistExample' as chosen dataset.
    """

    return {
        'x': {'Input': dataset['AETROS/dataset/VGG-16']['X_test']},
        'y': {'dense_3': dataset['AETROS/dataset/VGG-16']['Y_test']}
    }

def get_optimizer(trainer):
    """Returns the optimizer

    Parameters
    ----------
    trainer : dict
        trainer.set_status: Method allowing you to set the current job status
        logger: Logger with 'write' method allowing you to write into the log
        settings: dict - job settings
        job: dict - job with all information
    """

    import keras.optimizers

    optimizer = keras.optimizers.SGD(lr=0.01, decay=0.0001, momentum=0.001, nesterov=False)

    return optimizer

def get_loss(trainer):
    """Returns the optimizer

    Parameters
    ----------
    trainer : dict
        trainer.set_status: Method allowing you to set the current job status
        logger: Logger with 'write' method allowing you to write into the log
        settings: dict - job settings
        job: dict - job with all information
    """

    loss = {'dense_3': 'categorical_crossentropy'}
    return loss


def train(trainer, model, training_data, validation_data):

    """Returns the model (not build, not trained)

    Parameters
    ----------
    trainer : dict
        trainer.set_status: Method allowing you to set the current job status
        logger: Logger with 'write' method allowing you to write into the log
        settings: dict - job settings
        job: dict - job with all information
    """
    nb_epoch = trainer.settings['epochs']
    batch_size = trainer.settings['batchSize']

    if trainer.has_generator(training_data['x']):
        model.fit_generator(
            trainer.get_first_generator(training_data['x']),
            samples_per_epoch=trainer.samples_per_epoch,
            nb_val_samples=trainer.nb_val_samples,

            nb_epoch=nb_epoch,
            verbose=0,
            validation_data=trainer.get_first_generator(validation_data['x']),
            callbacks=trainer.callbacks
        )
    else:
        model.fit(
            training_data['x'],
            training_data['y'],
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            verbose=0,
            validation_data=(validation_data['x'], validation_data['y']),
            callbacks=trainer.callbacks
        )