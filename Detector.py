import tensorflow as tf
import numpy as np
import cv2
import utils
import time
from tensorflow.python.tools import inspect_checkpoint as chkp

class Detector():
    def __init__(self,sess= tf.InteractiveSession(),isRestore=False):
        self.sess=sess

        self.input_img = tf.placeholder(tf.float32, shape=(None, 416, 416, 3))
        # set some parameters
        self.net_h, self.net_w = 416, 416
        self.obj_thresh, self.nms_thresh = 0.5, 0.45
        self.anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
                  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
                  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
                  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
                  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
                  "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        self.logits=self.make_yolov3_model(self.input_img)

        tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo3/conv_block')).restore(
            self.sess, "./Weights/Yolov3.ckpt")
        print('success key model initialize')
    def _conv_block(self,inp, convs, skip=True,trainable=False):
        with tf.variable_scope('conv_block'):
            x = inp
            count = 0
            for conv in convs:
                if count == (len(convs) - 2) and skip:
                    skip_connection = x
                count += 1
                if conv['stride'] > 1:
                    paddings = tf.constant([[0,0],[1,0],[1,0],[0,0]])# peculiar padding as darknet prefer left and top
                    x=tf.pad(x,paddings)
                x=tf.layers.conv2d(inputs=x,
                                   filters=conv['filter'],
                                   kernel_size=[conv['kernel'],conv['kernel']],
                                   strides=(conv['stride'],conv['stride']),
                                   padding='valid' if conv['stride'] > 1 else 'same',
                                   name='conv_' + str(conv['layer_idx']),
                                   use_bias=False if conv['bnorm'] else True,
                                   trainable=trainable)
                if conv['bnorm']:
                    gamma=tf.get_variable(name='bnorm_'+str(conv['layer_idx'])+'/gamma',shape=(conv['filter']),trainable=trainable)
                    beta = tf.get_variable(name='bnorm_' + str(conv['layer_idx']) + '/beta', shape=(conv['filter']),trainable=trainable)
                    mean = tf.get_variable(name='bnorm_' + str(conv['layer_idx']) + '/moving_mean', shape=(conv['filter']),trainable=trainable)
                    variance = tf.get_variable(name='bnorm_' + str(conv['layer_idx']) + '/moving_variance', shape=(conv['filter']),trainable=trainable)
                    x = tf.nn.batch_normalization(x=x,mean=mean,variance=variance,scale=gamma,offset=beta,variance_epsilon=0.001)
                if conv['leaky']:
                    x = tf.nn.leaky_relu(x,alpha=0.1, name='leaky_' + str(conv['layer_idx']))
        return skip_connection+x if skip else x
    def make_yolov3_model(self,input_image,reuse=False):
        with tf.variable_scope('yolo3',reuse=reuse):
            # Layer  0 => 4
            x = self._conv_block(input_image,
                                 [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                   'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True,
                                   'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                                   'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                   'layer_idx': 3}])
            # Layer  5 => 8
            x = self._conv_block(x, [
                {'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
            # Layer  9 => 11
            x = self._conv_block(x, [
                {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])
            # Layer 12 => 15
            x = self._conv_block(x, [
                {'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])
            # Layer 16 => 36
            for i in range(7):
                x = self._conv_block(x, [
                    {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
                    {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17 + i * 3}])
            skip_36 = x
            # Layer 37 => 40
            x = self._conv_block(x, [
                {'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])
            # Layer 41 => 61
            for i in range(7):
                x = self._conv_block(x, [
                    {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
                    {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42 + i * 3}])
            skip_61 = x
            # Layer 62 => 65
            x = self._conv_block(x, [
                {'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])
            # Layer 66 => 74
            for i in range(3):
                x = self._conv_block(x, [
                    {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
                    {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67 + i * 3}])
            # Layer 75 => 79
            x = self._conv_block(x, [
                {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}],
                                 skip=False)
            # Layer 80 => 82
            yolo_82=self._conv_block(x, [
                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80},
                {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                 'layer_idx': 81}], skip=False)
            # Layer 83 => 86
            x = self._conv_block(x, [
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}],
                                 skip=False)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.concat([x, skip_61],axis=-1)
            # Layer 87 => 91
            x = self._conv_block(x, [
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}],
                                 skip=False)
            # Layer 92 => 94
            yolo_94=self._conv_block(x,
                                            [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                              'layer_idx': 92},
                                             {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                                              'layer_idx': 93}], skip=False)
            # Layer 95 => 98
            x = self._conv_block(x, [
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96}],
                                 skip=False)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.concat([x, skip_36],axis=-1)
            # Layer 99 => 106
            yolo_106=self._conv_block(x, [
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 99},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 100},
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 101},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 102},
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 103},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                 'layer_idx': 104},
                {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                 'layer_idx': 105}], skip=False)

        return [yolo_82,yolo_94,yolo_106]
    def Test(self,imgPath=None,img=None):
        # preprocess the image
        image = cv2.imread(imgPath)
        new_h, new_w, _ = image.shape
        # determine the new size of the image
        if (float(900) / new_w) < (float(900) / new_h):
            new_h = int((new_h * 900) / new_w)
            new_w = int(900)
        else:
            new_w = int((new_w * 900) / new_h)
            new_h = int(900)
        # resize the image to the new size
        image = cv2.resize(image, (new_w, new_h))
        image_h, image_w, _ = image.shape

        new_image = utils.preprocess_input(image, self.net_h, self.net_w)
        new_image = np.expand_dims(new_image, 0)

        #test=self.sess.run(self.feature_map_tensors[0],feed_dict=)
        # run the prediction
        yolos=self.sess.run(self.logits,feed_dict={self.input_img:new_image})

        boxes = []

        for i in range(len(yolos)):
            # decode the output of the network
            boxes += utils.decode_netout(yolos[i][0], self.anchors[i], self.obj_thresh, self.nms_thresh, self.net_h, self.net_w)

        # correct the sizes of the bounding boxes
        utils.correct_yolo_boxes(boxes, image_h, image_w, self.net_h, self.net_w)

        # suppress non-maximal boxes
        utils.do_nms(boxes, self.nms_thresh)

        # draw bounding boxes on the image using labels
        utils.draw_boxes(image, boxes, self.labels, self.obj_thresh)

        return image