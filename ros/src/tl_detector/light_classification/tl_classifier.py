from styx_msgs.msg import TrafficLight
from squeezenet import SqueezeNet
import tensorflow as tf
import numpy as np
import cv2

class TLClassifier:
    def __init__(self, is_site):
        #TODO load classifier
        assert not is_site
        weights_file = r'light_classification/models/squeezenet_weights.h5' #Replace with real world classifier
        
        image_shape = (224, 224, 3)

        self.states = (TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN, TrafficLight.UNKNOWN)

        print('Loading model..')
        self.model = SqueezeNet(len(self.states), *image_shape)
        self.model.load_weights(weights_file, by_name=True)
        self.model._make_predict_function()
        print('Loaded weights: %s' % weights_file)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        mini_batch = cv2.resize(image, (224,224), cv2.INTER_AREA).astype('float')[np.newaxis, ..., ::-1]/255.
        light = self.states[np.argmax(self.model.predict(mini_batch))]

        return light
