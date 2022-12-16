import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import utils
from config import cfg

anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
strides = np.array(cfg.YOLO.STRIDES)
anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
classes  = utils.read_class_names(cfg.YOLO.CLASSES)
num_class = len(classes)
iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        
def focal(target, actual, alpha=1, gamma=2):
    focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
    return focal_loss

def bbox_giou(boxes1, boxes2):


    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    #iou = inter_area / (union_area + 1e-5)
    iou = tf.math.divide_no_nan(inter_area, (union_area + 1e-5))
    # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    #giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-5)
    giou = iou - 1.0 * tf.math.divide_no_nan((enclose_area - union_area), (enclose_area + 1e-5))
    # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

    return giou

def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area

    return iou

def loss_layer(conv, pred, label, bboxes, anchors, stride):

    conv_shape  = conv.shape
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = stride * output_size

    conv = layers.Reshape((output_size, output_size,
                             anchor_per_scale, 5 + num_class))(conv)
    
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    print('pred_xywh shape: {}'.format(pred_xywh.shape))
    print('bboxes shape: {}'.format(bboxes.shape))
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < iou_loss_thresh, tf.float32 )

    conf_focal = focal(respond_bbox, pred_conf)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss



def custom_loss(label_sbbox, label_mbbox, label_lbbox, 
		true_sbbox, true_mbbox, true_lbbox,
		pred_sbbox, pred_mbbox, pred_lbbox,
		conv_sbbox, conv_mbbox, conv_lbbox
		): 
  
    #with tf.name_scope('smaller_box_loss'):
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                anchors = anchors[0], stride = strides[0])

    #with tf.name_scope('medium_box_loss'):
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                anchors = anchors[1], stride = strides[1])

    #with tf.name_scope('bigger_box_loss'):
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                anchors = anchors[2], stride = strides[2])

    #with tf.name_scope('giou_loss'):
    giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

    #with tf.name_scope('conf_loss'):
    conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

    #with tf.name_scope('prob_loss'):
    prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

    return giou_loss + conf_loss + prob_loss 



def loss_giou(label_sbbox, label_mbbox, label_lbbox, 
		true_sbbox, true_mbbox, true_lbbox,
		pred_sbbox, pred_mbbox, pred_lbbox,
		conv_sbbox, conv_mbbox, conv_lbbox
		):

  
    #with tf.name_scope('smaller_box_loss'):
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                anchors = anchors[0], stride = strides[0])

    #with tf.name_scope('medium_box_loss'):
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                anchors = anchors[1], stride = strides[1])

    #with tf.name_scope('bigger_box_loss'):
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                anchors = anchors[2], stride = strides[2])

    #with tf.name_scope('giou_loss'):
    giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

    return giou_loss 


def loss_conf(label_sbbox, label_mbbox, label_lbbox, 
		true_sbbox, true_mbbox, true_lbbox,
		pred_sbbox, pred_mbbox, pred_lbbox,
		conv_sbbox, conv_mbbox, conv_lbbox
		): 
  
    #with tf.name_scope('smaller_box_loss'):
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                anchors = anchors[0], stride = strides[0])

    #with tf.name_scope('medium_box_loss'):
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                anchors = anchors[1], stride = strides[1])

    #with tf.name_scope('bigger_box_loss'):
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                anchors = anchors[2], stride = strides[2])



    #with tf.name_scope('conf_loss'):
    conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]


    return conf_loss 




def loss_prob(label_sbbox, label_mbbox, label_lbbox, 
		true_sbbox, true_mbbox, true_lbbox,
		pred_sbbox, pred_mbbox, pred_lbbox,
		conv_sbbox, conv_mbbox, conv_lbbox
		): 
  
    #with tf.name_scope('smaller_box_loss'):
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                anchors = anchors[0], stride = strides[0])

    #with tf.name_scope('medium_box_loss'):
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                anchors = anchors[1], stride = strides[1])

    #with tf.name_scope('bigger_box_loss'):
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                anchors = anchors[2], stride = strides[2])

    #with tf.name_scope('prob_loss'):
    prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

    return prob_loss 
