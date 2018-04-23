import tensorflow as tf
import numpy as np
import cv2
import utils
import BatchGenerator
import time
from tensorflow.python.tools import inspect_checkpoint as chkp

class Detector():
    def __init__(self,sess= tf.InteractiveSession(),isRestore=False):
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
        self.batch_size=2
        self.min_net_size = 320
        self.max_net_size = 608
        self.ignore_thresh = 0.5
        self.warmup_batches = 3
        self.learning_rate=0.00001

        #self.anchors = tf.constant(anchors, dtype='float', shape=[3,1,1,1,3,2])

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.max_net_size), [self.max_net_size]), (1, self.max_net_size, self.max_net_size, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, 3, 1])


        self.sess=sess

        self.input_img = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.ground_truth = [tf.placeholder(tf.float32, shape=(None, None, None,3, len(self.labels)+5)),
                             tf.placeholder(tf.float32, shape=(None, None, None,3, len(self.labels)+5)),
                             tf.placeholder(tf.float32, shape=(None, None, None,3, len(self.labels)+5))]
        self.ground_truth_boxes = tf.placeholder(tf.float32, shape=(None, 1, 1, 1, None, 4))

        self.logits = self.make_yolov3_model(self.input_img)

        self.loss = tf.reduce_sum(self.LossFunction(self.input_img, self.logits, self.ground_truth, self.ground_truth_boxes))
        self.sqrt_loss = tf.sqrt(self.loss)


        train_ints, valid_ints, labels = utils.create_training_instances(
            'F:/DataSet/COCO/Annotations/instances_train2017.json',
            'F:/DataSet/COCO/train/',
            'coco_train_data',
            'F:/DataSet/COCO/Annotations/instances_val2017.json',
            'F:/DataSet/COCO/val/',
            'coco_val_data',
            self.labels
        )
        self.trainBatchGenerator = BatchGenerator.BatchGenerator(
            train_ints, self.anchors, self.labels,
            shuffle = True,
            jitter = 0.3,
            norm = utils.normalize,
            batch_size=self.batch_size
        )
        self.validBatchGenerator = BatchGenerator.BatchGenerator(
            valid_ints, self.anchors, self.labels,
            shuffle = True,
            jitter = 0.3,
            norm = utils.normalize,
            batch_size=self.batch_size)
        tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo3/conv_block')).restore(self.sess, "./Weights/Yolov3.ckpt")

        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.sqrt_loss)

        # List all global variables
        global_vars = tf.global_variables()

        # Find initialized status for all variables
        is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
        is_initialized = self.sess.run(is_var_init)

        # List all variables that were not initialized previously
        not_initialized_vars = [var for (var, init) in
                                zip(global_vars, is_initialized) if not init]

        # Initialize all uninitialized variables found, if any
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))
        print('success model initialize')
    def _conv_block(self,inp, convs, skip=True,trainable=True):
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
                    gamma=tf.get_variable(name='bnorm_'+str(conv['layer_idx'])+'/gamma',shape=(conv['filter']),trainable=False)
                    beta = tf.get_variable(name='bnorm_' + str(conv['layer_idx']) + '/beta', shape=(conv['filter']),trainable=False)
                    mean = tf.get_variable(name='bnorm_' + str(conv['layer_idx']) + '/moving_mean', shape=(conv['filter']),trainable=False)
                    variance = tf.get_variable(name='bnorm_' + str(conv['layer_idx']) + '/moving_variance', shape=(conv['filter']),trainable=False)
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
                {'filter': 3*(len(self.labels)+5), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
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
                                             {'filter': 3*(len(self.labels)+5), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
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
                {'filter': 3*(len(self.labels)+5), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                 'layer_idx': 105}], skip=False)

        return [yolo_82,yolo_94,yolo_106]
    def LossFunction(self, input_image, y_preds, y_trues, true_boxes):
        result_loss=[]
        for y_idx,y_pred in enumerate(y_preds):
            # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
            y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
            # initialize the masks
            object_mask = tf.expand_dims(y_trues[y_idx][..., 4], 4)
            no_object_mask = 1 - object_mask

            # the variable to keep track of number of batches processed
            batch_seen = tf.Variable(0.)

            # compute grid factor and net factor
            grid_h = tf.shape(y_trues[y_idx])[1]
            grid_w = tf.shape(y_trues[y_idx])[2]
            grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

            net_h = tf.shape(input_image)[1]
            net_w = tf.shape(input_image)[2]
            net_factor = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

            """
            Adjust prediction
            """
            pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
            pred_box_wh = y_pred[..., 2:4]  # t_wh
            pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)  # adjust confidence
            pred_box_class = tf.sigmoid(y_pred[..., 5:])  # adjust class probabilities

            """
            Adjust ground truth
            """
            true_box_xy = y_trues[y_idx][..., 0:2]  # (sigma(t_xy) + c_xy)
            true_box_wh = y_trues[y_idx][..., 2:4]  # t_wh
            true_box_conf = tf.expand_dims(y_trues[y_idx][..., 4], 4)
            true_box_class = y_trues[y_idx][..., 5:]

            """
            Compare each predicted box to all true boxes
            """
            # initially, drag all objectness of all boxes to 0
            conf_delta = pred_box_conf - 0

            # then, ignore the boxes which have good overlap with some true box
            true_xy = true_boxes[..., 0:2] / grid_factor
            true_wh = true_boxes[..., 2:4] / net_factor

            true_wh_half = true_wh / 2.
            true_mins = true_xy - true_wh_half
            true_maxes = true_xy + true_wh_half

            pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
            pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * np.reshape(self.anchors[y_idx],[1,1,1,3,2]) / net_factor, 4)

            pred_wh_half = pred_wh / 2.
            pred_mins = pred_xy - pred_wh_half
            pred_maxes = pred_xy + pred_wh_half

            intersect_mins = tf.maximum(pred_mins, true_mins)
            intersect_maxes = tf.minimum(pred_maxes, true_maxes)
            intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

            true_areas = true_wh[..., 0] * true_wh[..., 1]
            pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

            union_areas = pred_areas + true_areas - intersect_areas
            iou_scores = tf.truediv(intersect_areas, union_areas)

            best_ious = tf.reduce_max(iou_scores, axis=4)
            conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

            """
            Compute some online statistics
            """
            true_xy = true_box_xy / grid_factor
            true_wh = tf.exp(true_box_wh) * np.reshape(self.anchors[y_idx],[1,1,1,3,2]) / net_factor

            true_wh_half = true_wh / 2.
            true_mins = true_xy - true_wh_half
            true_maxes = true_xy + true_wh_half

            pred_xy = pred_box_xy / grid_factor
            pred_wh = tf.exp(pred_box_wh) * np.reshape(self.anchors[y_idx],[1,1,1,3,2]) / net_factor

            pred_wh_half = pred_wh / 2.
            pred_mins = pred_xy - pred_wh_half
            pred_maxes = pred_xy + pred_wh_half

            intersect_mins = tf.maximum(pred_mins, true_mins)
            intersect_maxes = tf.minimum(pred_maxes, true_maxes)
            intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

            true_areas = true_wh[..., 0] * true_wh[..., 1]
            pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

            union_areas = pred_areas + true_areas - intersect_areas
            iou_scores = tf.truediv(intersect_areas, union_areas)
            iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

            count = tf.reduce_sum(tf.to_float(object_mask))
            count_noobj = tf.reduce_sum(tf.to_float(no_object_mask))
            recall50 = tf.reduce_sum(tf.to_float(iou_scores >= 0.5)) / (count + 1e-3)
            recall75 = tf.reduce_sum(tf.to_float(iou_scores >= 0.75)) / (count + 1e-3)
            avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
            avg_obj = tf.reduce_sum(object_mask * pred_box_conf * true_box_conf) / (count + 1e-3)
            avg_noobj = tf.reduce_sum(no_object_mask * pred_box_conf) / (count_noobj + 1e-3)
            avg_cat = tf.reduce_sum(object_mask * pred_box_class * true_box_class) / (count + 1e-3)

            """
            Warm-up training
            """
            batch_seen = tf.assign_add(batch_seen, 1.)

            true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches + 1),
                                                          lambda: [true_box_xy + (
                                                                  0.5 + self.cell_grid[:, :grid_h, :grid_w, :,
                                                                        :]) * no_object_mask,
                                                                   true_box_wh + tf.zeros_like(
                                                                       true_box_wh) * no_object_mask,
                                                                   tf.ones_like(object_mask)],
                                                          lambda: [true_box_xy,
                                                                   true_box_wh,
                                                                   object_mask])
            """
            Compare each true box to all anchor boxes
            """
            xywh_scale = tf.exp(true_box_wh) * np.reshape(self.anchors[y_idx],[1,1,1,3,2]) / net_factor
            xywh_scale = tf.expand_dims(2 - xywh_scale[..., 0] * xywh_scale[..., 1],
                                        axis=4)  # the smaller the box, the bigger the scale

            xy_delta = xywh_mask * (pred_box_xy - true_box_xy) * xywh_scale
            wh_delta = xywh_mask * (pred_box_wh - true_box_wh) * xywh_scale
            conf_delta = object_mask * (pred_box_conf - true_box_conf) * 5 + (1 - object_mask) * conf_delta
            class_delta = object_mask * (pred_box_class - true_box_class)

            loss = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5))) + \
                   tf.reduce_sum(tf.square(wh_delta), list(range(1, 5))) + \
                   tf.reduce_sum(tf.square(conf_delta), list(range(1, 5))) + \
                   tf.reduce_sum(tf.square(class_delta), list(range(1, 5)))

            loss = tf.cond(tf.less(batch_seen, self.warmup_batches + 1),  # add 10 to the loss if this is the warmup stage
                           lambda: loss + 10,
                           lambda: loss)

            # loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
            # loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
            # loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
            # loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
            # loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
            # loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
            # loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
            # loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss)], message='loss: \t', summarize=1000)
            result_loss.append(loss)

        return result_loss
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
    def Training(self):
        for epoch in range(10):
            loss_sum=0
            for iter in range(self.trainBatchGenerator.__len__()):
                train_batch = self.trainBatchGenerator.__getitem__(0)
                feed_dict = {self.input_img: train_batch[0], self.ground_truth_boxes: train_batch[1],
                             self.ground_truth[0]: train_batch[2], self.ground_truth[1]: train_batch[3],
                             self.ground_truth[2]: train_batch[4]}

                loss,_=self.sess.run([self.loss,self.train_op],feed_dict=feed_dict)
                loss_sum+=loss
                if (iter%100==0):
                    print('iter = ',iter,' loss = ',loss)

            tf.train.Saver().save(self.sess, "./Weights/test_"+str(epoch)+".ckpt")