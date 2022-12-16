# yolov3-tf2
An implementation of yolov3 using tensorflow 2 functional API. 

This repository is adapted from the following implementation by YunYang1994: https://github.com/YunYang1994/tensorflow-yolov3

This code is rewritten using the functional API enabling eager execution, making it more compatible with tf2 as compared to the original repository. (For the original repository you need to disable eager execution and manually add 'compat.v1' to a large number of functions)

This project uses an example of detecting liver lesions in 2d abdominal CT images.

To adapt this project for your own uses:

Change data/dataset/liver_train.txt, data/dataset/liver_test.txt into your own training and testing annotations. The format of annotations is the same as the original repository.
Change how the data is loaded, in dataset.py, function parse_annotation(), line 233. This example uses .npy files, so the code to load the image is np.load().
Change data/classes/liver.names to define your classes.
