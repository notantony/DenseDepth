import os
import glob
import matplotlib
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from io import BytesIO
from PIL import Image
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, save_images, DepthNorm
from matplotlib import pyplot as plt
from server.api.manager import ModelManager, Payload


# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}


class Estimator():
    # min/max in centimeters
    def __init__(self, model_path, min_depth=10, max_depth=1000):
        print('Loading model...')

        self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.manager = ModelManager()
        self.min_depth = min_depth
        self.max_depth = max_depth

        print('Model loaded: {}'.format(model_path))


    def __enter__(self):
        self.manager.__enter__()
        return self


    def __exit__(self, type, value, traceback):
        self.manager.__exit__(type, value, traceback)


    def _estimate(self, inputs):
        with self.graph.as_default():
            outputs = self.model.predict(inputs)

        return outputs


    def estimate(self, image_data):
        image = Image.open(BytesIO(image_data))
        if image.size != (640, 480):
            image = image.resize((640, 480), Image.BICUBIC)
        image = image.convert('RGB')
        input_data = np.clip(np.asarray(image, dtype=float) / 255, 0, 1)
        x = np.expand_dims(input_data, axis=0)
        inputs = x
        
        payload = Payload(self._estimate, (inputs,))

        predictions = self.manager.send_task(payload)
        outputs = np.clip(DepthNorm(predictions, maxDepth=self.max_depth), self.min_depth, self.max_depth) / self.max_depth

        return np.squeeze(outputs[0])


    def debug(self, inputs, outputs):
        print(outputs.shape)
        print(np.squeeze(outputs[0]))
        save_images("./tmp_rescaled.png", outputs.copy(), inputs.copy(), is_colormap=True, is_rescale=True)
        save_images("./tmp_dm.png", outputs.copy(), inputs.copy(), is_colormap=True, is_rescale=False)
