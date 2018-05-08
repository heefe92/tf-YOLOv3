import cv2
import numpy as np
import copy
import os
import pickle
import xml.etree.ElementTree as ET
import json
import struct
import tensorflow as tf

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = int(i / grid_w)
        col = int(i % grid_w)

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if (objectness <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row, col, b, :4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[row, col, b, 5:]

            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)

            boxes.append(box)

    return boxes
def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = int((new_h * net_w) / new_w)
        new_w = int(net_w)
    else:
        new_w = int((new_w * net_h) / new_h)
        new_h = int(net_h)
    # resize the image to the new size
    resized = cv2.resize(image / 255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5


    new_image[ int((net_h - new_h) / 2):int((net_h + new_h) / 2), int((net_w - new_w) / 2):int((net_w + new_w) / 2), :] = resized[:,:,:]


    return new_image
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
def draw_boxes(image, boxes, labels, obj_thresh):
    colors = [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127),
              (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
              (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
              (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
              (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254),
              (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
              (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127), (254.0, 254.0, 254),
              (239.88888888888889, 211.66666666666669, 127),
              (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
              (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
              (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
              (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254),
              (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
              (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127), (254.0, 254.0, 254),
              (239.88888888888889, 211.66666666666669, 127),
              (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
              (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
              (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
              (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254),
              (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
              (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127), (254.0, 254.0, 254),
              (239.88888888888889, 211.66666666666669, 127),
              (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
              (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
              (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
              (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254),
              (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
              (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]

    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                #print(labels[i] + ': ' + str(box.classes[i] * 100) + '%')

        if label >= 0:
            cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), colors[label], 3)
            cv2.putText(image,
                        label_str + ' ' + str(box.get_score()),
                        (box.xmin, box.ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        colors[label], 2)
    return image
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect
    if(union==0):
        return 0
    return float(intersect) / union
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)

            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def load_weights(self, model,sess):
        for i in range(106):
            try:
                conv_layer=tf.get_collection(model, scope='yolo3/conv_block/conv_'+str(i)+'/')
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = tf.get_collection(model, scope='yolo3/conv_block/bnorm_'+str(i)+'/')

                    size = np.prod(norm_layer[0].shape)

                    beta = self.read_bytes(size)  # bias
                    gamma = self.read_bytes(size)  # scale
                    mean = self.read_bytes(size)  # mean
                    var = self.read_bytes(size)  # variance
                    sess.run(tf.assign(norm_layer[0],gamma))
                    sess.run(tf.assign(norm_layer[1], beta))
                    sess.run(tf.assign(norm_layer[2], mean))
                    sess.run(tf.assign(norm_layer[3], var))

                if len(conv_layer) > 1:
                    bias = self.read_bytes(np.prod(conv_layer[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer[0].shape))

                    kernel = kernel.reshape(list(reversed(conv_layer[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    sess.run(tf.assign(conv_layer[0], kernel))
                    sess.run(tf.assign(conv_layer[1], bias))
                else:
                    kernel = self.read_bytes(np.prod(conv_layer[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    sess.run(tf.assign(conv_layer[0], kernel))
            except Exception:
                print("no convolution #" + str(i))

    def reset(self):
        self.offset = 0

def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1. / scale;
def _constrain(min_v, max_v, value):
    if value < min_v: return min_v
    if value > max_v: return max_v
    return value
def random_flip(image, flip):
    if flip == 1: return cv2.flip(image, 1)
    return image
def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w) / image_w, float(new_h) / image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin'] * sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax'] * sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin'] * sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax'] * sy + dy))

        if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
            zero_boxes += [i]
            continue

        if flip == 1:
            swap = boxes[i]['xmin'];
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes
def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation);
    dexp = _rand_scale(exposure);

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp

    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180

    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)
def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    im_sized = cv2.resize(image, (new_w, new_h))

    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (dx, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:, -dx:, :]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0, 0), (0, net_w - (new_w + dx)), (0, 0)), mode='constant', constant_values=127)

    if dy > 0:
        im_sized = np.pad(im_sized, ((dy, 0), (0, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:, :, :]

    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h + dy)), (0, 0), (0, 0)), mode='constant', constant_values=127)

    return im_sized[:net_h, :net_w, :]


def parse_coco_annotation(ann_file, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']

    else:
        all_insts = []
        seen_labels = {}
        ann_data=json.load(open(ann_file))
        categories=ann_data['categories']
        labels_id=[]
        print(len(ann_data['annotations']))
        for category in categories:
            if category['name'] in labels:
                labels_id.append(category['id'])
        for idx,ann in enumerate(ann_data['annotations']):
            if(idx%1000==0):
                print(idx)
            if ann['category_id'] in labels_id:
                img_id=str(ann['image_id'])
                while len(img_id)<12:
                    img_id='0'+img_id
                file_name=img_dir+img_id+'.jpg'
                flag=True
                for inst in all_insts:
                    if(inst['filename']==file_name):
                        object = {}
                        object['name'] = labels[labels_id.index(ann['category_id'])]
                        object['xmin'] = ann['bbox'][0]
                        object['ymin'] = ann['bbox'][1]
                        object['xmax'] = int(ann['bbox'][0]+ann['bbox'][2]) - 1
                        object['ymax'] = int(ann['bbox'][1]+ann['bbox'][3]) - 1
                        inst['object'].append(object)
                        if object['name'] in seen_labels:
                            seen_labels[object['name']] += 1
                        else:
                            seen_labels[object['name']] = 1
                        flag=False
                if flag:
                    instance={}
                    instance['filename'] = file_name
                    img=cv2.imread(file_name)
                    instance['height'] = img.shape[0]
                    instance['width'] = img.shape[1]
                    instance['object']=[]
                    object={}

                    object['name']=labels[labels_id.index(ann['category_id'])]
                    object['xmin']= int(ann['bbox'][0])
                    object['ymin'] = int(ann['bbox'][1])
                    object['xmax'] = int(ann['bbox'][0]+ann['bbox'][2]) - 1
                    object['ymax'] = int(ann['bbox'][1]+ann['bbox'][3]) - 1
                    instance['object'].append(object)
                    if object['name'] in seen_labels:
                        seen_labels[object['name']] += 1
                    else:
                        seen_labels[object['name']] = 1
                    all_insts.append(instance)

                #print(str(ann['image_id']))

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels

def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_coco_annotation(train_annot_folder, train_image_folder, train_cache, labels)
    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_coco_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.shuffle(train_ints)

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t\t'  + str(train_labels))
        print('Given labels: \t\t' + str(labels))
        print('Overlap labels: \t' + str(list(overlap_labels)))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            # return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    return train_ints, valid_ints, sorted(labels)
def normalize(image):
    return image/255.