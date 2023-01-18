
import tensorflow as tf
import pickle


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)


class DigitModel(object):

    def __init__(self, model_file):
        with open(model_file, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def predict_digit(self, img):
        self.preds = self.model.predict(img)
        return self.preds
