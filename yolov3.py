import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from common import *
from backbone import *
import utils
from config import cfg
from losses import *

anchors_path = ''
iou_loss_thresh = 0.5
 

           
def YoloV3(training=True):

    ##########Functions for loss calculations##########     
    def decode(conv_output, anchors, stride):
        conv_shape = conv_output.shape
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchors_per_scale = len(anchors)
        #print('conv_output shape: {}'.format(conv_output.shape))

        conv_output = layers.Reshape(
            (output_size, output_size,
             anchors_per_scale,
             5 + num_class))(conv_output)

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        print('y shape: {}'.format(y.shape))
        print('x shape: {}'.format(x.shape))
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        print('xy_grid shape: {}'.format(xy_grid.shape))
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchors_per_scale, 1])
        #xy_grid = tf.cast(xy_grid, tf.float32)
        xy_grid = layers.Lambda(lambda x: tf.cast(x, tf.float32))(xy_grid)
        
        print('xy_grid shape: {}'.format(xy_grid.shape))
        print('conv_raw_dxdy shape: {}'.format(conv_raw_dxdy.shape))
        print('anchors shape: {}'.format(anchors.shape))
        print('stride shape: {}'.format(stride.shape))
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    
    
    #Model parameters
    factor             = cfg.YOLO.FACTOR
    trainable          = training
    classes            = utils.read_class_names(cfg.YOLO.CLASSES)
    num_class          = len(classes)
    strides            = np.array(cfg.YOLO.STRIDES)
    anchors            = utils.get_anchors(cfg.YOLO.ANCHORS)
    anchors_per_scale   = cfg.YOLO.ANCHOR_PER_SCALE
    iou_loss_thresh    = cfg.YOLO.IOU_LOSS_THRESH
    upsample_method    = cfg.YOLO.UPSAMPLE_METHOD
    batch_size         = cfg.YOLO.BATCH_SIZE
    input_size         = cfg.YOLO.INPUT_SIZE
    max_bbox_per_scale = cfg.YOLO.MAX_BBOX_PER_SCALE
    

    #Input data
    input_data_org = layers.Input(shape=(input_size, input_size, 3), batch_size=batch_size,
			      name = 'input_data')
    
    label_sbbox = layers.Input(shape=(input_size // strides[0],
                                      input_size // strides[0],
                                      anchors_per_scale,
                                      5 + num_class), batch_size=batch_size,
                                      name = 'input_label_sbbox')
    label_mbbox = layers.Input(shape=(input_size // strides[1],
                                      input_size // strides[1],
                                      anchors_per_scale,
                                      5 + num_class), batch_size=batch_size,
                                      name = 'input_label_mbbox'
                                      )        
    label_lbbox = layers.Input(shape=(input_size // strides[2],
                                      input_size // strides[2],
                                      anchors_per_scale,
                                      5 + num_class), batch_size=batch_size,
                                      name = 'input_label_lbbox')

    true_sbboxes = layers.Input(shape=(max_bbox_per_scale, 4),
                                batch_size=batch_size,
                                name = 'input_true_sbboxes')
    true_mbboxes = layers.Input(shape=(max_bbox_per_scale, 4),
                                batch_size=batch_size,
                                name = 'input_true_mbboxes')
    true_lbboxes = layers.Input(shape=(max_bbox_per_scale, 4),
                                batch_size=batch_size,
                                name = 'input_true_lbboxes')
    
    ###Pass through darknet53###
    route1, route2, input_data = darknet53(input_data_org)

    ###Three branches to output lobj, mobj, sobj###
    #
    input_data = conv_series(input_data, [(1,1), (3,3), (1,1), (3,3), (1,1)],
                             [factor*32, factor*64, factor*32, factor*64, factor*32],
                             name = 'lobj')
    conv_lobj_branch = conv_block(input_data, (3, 3), factor*64, trainable=trainable,
                                  name = 'conv_lobj_branch')
    conv_lbbox = conv_block(conv_lobj_branch, (1,1), 3*(num_class + 5),
                            trainable=trainable, name='conv_lbbox', activate=False, bn=False)
    #
    input_data = conv_block(input_data, (1, 1), factor*16, trainable=trainable,
                            name = 'up1_conv')
    input_data = upsample_block(input_data, name = 'upsample1')

    #print('input_data shape: {}'.format(input_data.shape))
    #print('route2 shape: {}'.format(route2.shape))
    input_data = layers.Concatenate(axis=-1)([input_data, route2])
    input_data = conv_series(input_data, [(1,1), (3,3), (1,1), (3,3), (1,1)],
                             [factor*16, factor*32, factor*16, factor*32, factor*16],
                             name = 'mobj') 
    conv_mobj_branch = conv_block(input_data, (3, 3), factor*32, trainable=trainable,
                                  name = 'conv_mobj_branch')
    conv_mbbox = conv_block(conv_mobj_branch, (1,1), 3*(num_class + 5),
                            trainable=trainable, name='conv_mbbox', activate=False, bn=False)
    
    #
    input_data = conv_block(input_data, (1, 1), factor*16, trainable=trainable,
                            name = 'up2_conv')
    input_data = upsample_block(input_data, name = 'upsample2')

    input_data = layers.Concatenate()([input_data, route1])
    input_data = conv_series(input_data, [(1,1), (3,3), (1,1), (3,3), (1,1)],
                             [factor*8, factor*16, factor*8, factor*16, factor*8],
                             name = 'sobj') 
    conv_sobj_branch = conv_block(input_data, (3, 3), factor*16, trainable=trainable,
                                  name = 'conv_sobj_branch')
    conv_sbbox = conv_block(conv_sobj_branch, (1,1), 3*(num_class + 5),
                            trainable=trainable, name='conv_sbbox', activate=False, bn=False)

    #Decode output to obtain predictions
    #print("anchors0: {}".format(anchors[0].shape))
    #print("anchors1: {}".format(anchors[1].shape))
    #print("anchors2: {}".format(anchors[2].shape))
    pred_sbbox = decode(conv_sbbox, anchors[0], strides[0])
    pred_mbbox = decode(conv_mbbox, anchors[1], strides[1])
    pred_lbbox = decode(conv_lbbox, anchors[2], strides[2])

    '''
    print('conv_sbbox shape: {}'.format(conv_sbbox.shape))
    print('conv_mbbox shape: {}'.format(conv_mbbox.shape))
    print('conv_lbbox shape: {}'.format(conv_lbbox.shape))
    
    print('pred_sbbox shape: {}'.format(pred_sbbox.shape))
    print('pred_mbbox shape: {}'.format(pred_mbbox.shape))
    print('pred_lbbox shape: {}'.format(pred_lbbox.shape))
    
    
    print('label_sbbox shape: {}'.format(label_sbbox.shape))
    print('label_mbbox shape: {}'.format(label_mbbox.shape))
    print('label_lbbox shape: {}'.format(label_lbbox.shape))

    print('true_sbboxes shape: {}'.format(true_sbboxes.shape))
    print('true_mbboxes shape: {}'.format(true_mbboxes.shape))
    print('true_lbboxes shape: {}'.format(true_lbboxes.shape))
    '''
    
    model = Model(#inputs=input_data_org,
		  inputs=[input_data_org, label_sbbox, label_mbbox, label_lbbox,
	                  true_sbboxes, true_mbboxes, true_lbboxes],
                  outputs=[pred_sbbox, pred_mbbox, pred_lbbox,
                           conv_sbbox, conv_mbbox, conv_lbbox],
                  name = 'YoloV3')
    loss = custom_loss(label_sbbox, label_mbbox, label_lbbox,
		        true_sbboxes, true_mbboxes, true_lbboxes,
		        pred_sbbox, pred_mbbox, pred_lbbox,
		        conv_sbbox, conv_mbbox, conv_lbbox)

    giou_loss = loss_giou(label_sbbox, label_mbbox, label_lbbox,
		        true_sbboxes, true_mbboxes, true_lbboxes,
		        pred_sbbox, pred_mbbox, pred_lbbox,
		        conv_sbbox, conv_mbbox, conv_lbbox)
    conf_loss = loss_conf(label_sbbox, label_mbbox, label_lbbox,
		        true_sbboxes, true_mbboxes, true_lbboxes,
		        pred_sbbox, pred_mbbox, pred_lbbox,
		        conv_sbbox, conv_mbbox, conv_lbbox)
    prob_loss = loss_prob(label_sbbox, label_mbbox, label_lbbox,
		        true_sbboxes, true_mbboxes, true_lbboxes,
		        pred_sbbox, pred_mbbox, pred_lbbox,
		        conv_sbbox, conv_mbbox, conv_lbbox)
    
    model.add_loss(loss)
    
    model.add_metric(giou_loss, name='loss_giou')
    model.add_metric(conf_loss, name='loss_conf')
    model.add_metric(prob_loss, name='loss_prob')
    
    return model

    

    
