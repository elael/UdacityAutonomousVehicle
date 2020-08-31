from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        if not is_site:
            path = r'light_classification/models/frozen_inference_graph_sim.pb'
        else:
            path = r'light_classification/models/frozen_inference_graph_sim.pb' #Replace with real world classifier
            
        self.graph = tf.Graph()
        # Min threshold for considering the classification reliable
        self.threshold = .5

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if scores[0] > self.threshold:
            if classes[0] == 1:
                print('GREEN')
                return TrafficLight.GREEN
            elif classes[0] == 2:
                print('RED')
                return TrafficLight.RED
            elif classes[0] == 3:
                print('YELLOW')
                return TrafficLight.YELLOW

        print('UNKNOWN')
        return TrafficLight.UNKNOWN
