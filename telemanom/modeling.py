import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.keras.callbacks import History, EarlyStopping
import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import logging

#from telemanom.utility import create_lstm_model, create_esn_model, create_path
from telemanom.utility import create_esn_model, create_path
import random
import time

# suppress tensorflow CPU speedup warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')

def get_seed(config, chan_id):
    """
    Read previously saved seeds
    :param folder: folder for seeds file
    :param chan_id: telemetry channel name
    :return: returns the seed saved for a given telemetry channel
    """

    path = create_path(config, 'models')+'seeds.log'
    #path = f'./data/{folder}/models/seeds.log'
    file1 = open(path, 'r')

    for row in file1.readlines():
        if row.startswith(chan_id):
            seed = int(row.strip().split(" ")[1])
            return seed


class Model:
    def __init__(self, config, run_id, channel):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.size = 0
        self.prediction_time = 0
        self.channel = channel

        if self.config.execution == "predict":
            try:
                logger.info('Loading pre-trained model')

                if self.config.model_architecture != "LSTM":
                    print('ESN')
                    hp = {}
                    if self.config.load_hp:
                        logger.info('Loading hp id: {}'.format(self.config.hp_research_id))
                        path = "./hp/{}/config/{}.yaml".format(self.config.hp_research_id, self.chan_id)
                        with open(path, 'r') as file:
                            hp = yaml.load(file, Loader=yaml.BaseLoader)

                    path = create_path(self.config, 'models')+self.chan_id
                    logger.info('Recreate model')
                    # get seed for a specific model model
                    seed = get_seed(self.config, self.chan_id)
                    self.model = create_esn_model(channel, self.config, hp, seed)
                    weight_path = path+'.h5'
                    self.model.load_weights(weight_path)

                    self.model._set_inputs(channel.X_test[:config.batch_size])
                    #to avoid E:ValueError: Model <telemanom.ESN.SimpleESN ... > cannot be saved because the input shapes have not been set. Usually, input shapes are automatically determined from calling `.fit()` or `.predict()`. To manually set the shapes, call `model.build(input_shape)`.
                    tf.saved_model.save(self.model, path)
                    self.size = os.path.getsize(path+'/saved_model.pb') / 1024
                else:
                    print('LSTM')
                    path = create_path(self.config, 'models')+self.chan_id+'.h5'
                    self.model = load_model(path)
                    print(self.model.summary())
                    self.size = os.path.getsize(path) / 1024


            except (FileNotFoundError, OSError) as e:
                print(e)
                path = os.path.join('data', self.config.use_id, 'models',
                                    self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(path))

                self.train_new(channel)
                self.save()

        elif self.config.execution == "train" or self.config.execution == "train_and_predict":
            self.train_new(channel)
            self.save()

        else:
            logger.info("Configuration file error, check execution flag")
            sys.exit("Configuration file error, check execution flag")

    def train_new(self, channel):
        """
        Train ESN or LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """
        pass
        """
            hp = {}
            if self.config.load_hp:
                path = os.path.join("hp", self.config.hp_research_id, "config", "{}.yaml".format(self.chan_id))
                try:
                    with open(path, 'r') as file:
                        hp = yaml.load(file, Loader=yaml.BaseLoader)
    
                    if self.config.model_architecture == "ESN":
                        logger.info('units: {}'.format(hp["units"]))
                        logger.info('input_scaling: {}'.format(hp["input_scaling"]))
                        logger.info('radius: {}'.format(hp["radius"]))
                        logger.info('leaky: {}'.format(hp["leaky"]))
                        logger.info('learning_rate: {}'.format(hp["learning_rate"]))
    
                    if self.config.model_architecture == "LSTM":
                        logger.info('units: {}'.format(hp["units"]))
                        logger.info('dropout: {}'.format(hp["dropout"]))
                        logger.info('learning_rate: {}'.format(hp["learning_rate"]))
                        logger.info('layers: {}'.format(hp["layers"]))
    
                except FileNotFoundError as e:
                    logger.info("No configuration file at {} using default hypeparameters".format(path))
                    raise e
            else:
                logger.info("default hp")
    
    
            cbs = [History(), EarlyStopping(monitor='val_loss',
                                            patience=self.config.patience,
                                            min_delta=self.config.min_delta,
                                            verbose=0)]
    
            if self.config.model_architecture == "LSTM":
    
                self.model = create_lstm_model(channel,self.config, hp)
    
    
                self.history = self.model.fit(channel.X_train,
                                              channel.y_train,
                                              batch_size=self.config.lstm_batch_size,
                                              epochs=self.config.epochs,
                                              validation_data=(channel.X_valid, channel.y_valid),
                                              callbacks=cbs,
                                              verbose=True)
    
            #esn
            else:
                SEED = random.randint(43, 999999999)
                if self.config.model_architecture == "ESN":
                    self.model = create_esn_model(channel,self.config, hp, SEED)
                    if self.config.serialization:
                        self.history = self.model.fit(channel.X_train,
                                                  channel.y_train,
                                                  validation_data=(channel.X_valid, channel.y_valid),
                                                  epochs=self.config.epochs,
                                                  callbacks=cbs,
                                                  verbose=True)
                    else:
                        self.history = self.model.fit(channel.X_train,
                                                      channel.y_train,
                                                      validation_data=(channel.X_valid, channel.y_valid),
                                                      epochs=self.config.epochs,
                                                      batch_size=self.config.lstm_batch_size,
                                                      callbacks=cbs,
                                                      verbose=True)
    
    
            logger.info('validation_loss: {}\n'.format(self.history.history["val_loss"][-1]))
        """

    def save(self):
        """
        Save trained model, loss and validation loss graphs .
        """
        pass
        """
            if self.config.save_graphs:
                plt.figure()
                plt.plot(self.history.history["loss"], label="Training Loss")
                plt.plot(self.history.history["val_loss"], label="Validation Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title(f'Training and validation loss model: {self.config.model_architecture} channel: {self.chan_id}')
    
                plt.legend()
    
                plt.savefig(os.path.join('data', self.run_id, 'images',
                                         '{}_loss.png'.format(self.chan_id)))
                #plt.show()
                plt.close()
    
            if self.config.model_architecture != "LSTM":
                self.model.save_weights(os.path.join('data', self.run_id, 'models',
                                             '{}.h5'.format(self.chan_id)))
                #saving seeds
                path = './data/{}/models/seeds.log'.format(self.run_id)
                f = open(path, "a")
                f.write("{} {}\n".format(self.chan_id, self.model.SEED))
                f.close()
    
    
            else:
                self.model.save(os.path.join('data', self.run_id, 'models',
                                             '{}.h5'.format(self.chan_id)))
        """

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """
        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel):
        """
        Used trained LSTM or ESN model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                          / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_test.shape[0]))

        n = channel.X_test.shape[0]
        start_time = time.time()
        method = self.config.method
        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                idx = channel.y_test.shape[0]

            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.model.predict(X_test_batch)
            self.aggregate_predictions(y_hat_batch, method=method)
            #progress
            percentage = int((idx * 100)/ n)
            sys.stdout.write('\r')
            sys.stdout.write(str(percentage)+'%')
            sys.stdout.flush()

        sys.stdout.write('\n')
        delta_time = time.time() - start_time
        self.prediction_time = delta_time
        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        channel.y_hat = self.y_hat
        path = create_path(self.config, 'y_hat') + self.chan_id + '.npy'
        np.save(path, self.y_hat)

        return channel


class LiteModel:

    def __init__(self, config, channel):
        """
            Convert/Loads TensorFlowLite models and predicts future telemetry values for a channel.

            Args:
                config (obj): Config object containing parameters for processing
                    and model training
                channel (obj): Channel class object containing train/test data for X,y for a single channel

            Attributes:
                config (obj): see Args
                chan_id (str): channel id
                y_hat (arr): predicted channel values
                model (obj): trained RNN model for predicting channel values using TensorFlow Lite
                size (int): model size (in KB)
                conversion_time (float): time elapled during conversion from TensorFlow to TensorFlow Lite
        """

        self.config = config
        self.chan_id = channel.id
        self.y_hat = np.array([])
        self.model = None
        self.model_path = create_path(self.config, 'models', lib='TFLite')+self.chan_id+'.tflite'
        self.size = 0
        self.conversion_time = 0
        self.prediction_time = 0

    def convert(self, tf_model, save=True):
        """
            Convert TensorFlow model in TensorFlowLite model

            Args:
                tf_model (obj): Model class object containing TensorFlow trained model
        """

        start_time = time.time()
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        #converter.allow_custom_ops = True
        #converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.model = converter.convert()
        delta_time = time.time() - start_time
        self.conversion_time = delta_time
        print('Model converted in {}s'.format(self.conversion_time))

        self.size = self.model.__sizeof__() / 1024

        # save TFLite model
        if save:
            with open(self.model_path, 'wb') as f:
                f.write(self.model)


    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """
        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel, save=True):

        """
        Used trained lite model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
            :param save: save predictions
        """

        num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                          / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_test.shape[0]))

        method = self.config.method

        #load TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # preprocessing input data
        input_shape = input_details[0]['shape']  # [1, 1, 25]
        output_shape = output_details[0]['shape']  # [1, 10]

        start_time = time.time()
        n = channel.X_test.shape[0]

        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                idx = channel.y_test.shape[0]

            # risetto la dimensione dell'input in modo da prendere un batch alla volta
            interpreter.resize_tensor_input(input_details[0]['index'], [idx-prior_idx, channel.X_test.shape[1], channel.X_test.shape[2]])
            interpreter.allocate_tensors()

            X_test_batch = channel.X_test[prior_idx:idx]

            #TODO- why lstm requires type= np.float32; esn requires type= np.float64 ?
            if self.config.model_architecture == 'LSTM':
                X_test_batch = np.array(X_test_batch, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], X_test_batch)
            interpreter.invoke()

            y_hat_batch = interpreter.get_tensor(output_details[0]['index'])
            #print(y_hat_batch)
            self.aggregate_predictions(y_hat_batch, method=method)
            # progress
            percentage = int((idx * 100) / n)
            sys.stdout.write('\r')
            sys.stdout.write(str(percentage) + '%')
            sys.stdout.flush()

        sys.stdout.write('\n')
        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))
        delta_time = time.time() - start_time
        self.prediction_time = delta_time
        print(self.prediction_time)
        channel.y_hat = self.y_hat
        if save:
            np.save(create_path(self.config, 'y_hat', lib='TFLite')+self.chan_id+'.npy', self.y_hat)

        return channel