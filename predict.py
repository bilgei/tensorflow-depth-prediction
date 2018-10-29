import argparse
import os
import numpy as np
import numpy.random as npr
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import models
import cv2
from plyfile import PlyData, PlyElement
import pointcloudfromdepth as pc
import nyu2_dataset as nyu
import losses

HEIGHT = 228
WIDTH = 304
CHANNELS = 3
BATCH_SIZE = 1

def construct_network():
    ## Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, CHANNELS))
    ## Construct the network  
    net = models.ResNet50UpProj({'data': input_node}, BATCH_SIZE, 1, False)
    return net, input_node


def predict(model_data_path, image_path, net, input_node, prev_pred):
    # Read image
    img = Image.open(image_path)
    img = img.resize([WIDTH, HEIGHT], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)

    print("img_shape: ", img.shape)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})

        ## Plot result
        #fig = plt.figure()
        #ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        #fig.colorbar(ii)
        #plt.show()
        #fig.savefig('depth.png', bbox_inches='tight')
        
        new_pred = pred[0,:,:,:]
        if prev_pred is None:
            prev_pred = new_pred
        
        prediction = np.median([new_pred, prev_pred], axis=0)

        return img, prediction

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()
 
    dataset = nyu.matchInputOutputPaths(args.image_paths)
    print("Dataset image count: ", len(dataset))
    net, input_node = construct_network()
    model_path = args.model_path

    total_metric = 0
    
    for i, sample in enumerate(dataset):
        image_path = sample[0]
        gt_path = sample[1]

        print("image_path: ", image_path)
        print("gt_path: ", gt_path)

        #if i == 0:
        prev_pred = None
        # Predict the image        
        rgb, pred = predict(model_path, image_path, net, input_node, prev_pred)
        prev_pred = pred

        imgDepthAbs = pred
        imgDepthAbs = cv2.resize(imgDepthAbs, (640,480))
        rgb = cv2.resize(rgb[0,:,:,:], (640,480))####    
        print(imgDepthAbs.shape)
        [H, W] = imgDepthAbs.shape
        print(imgDepthAbs.max(), imgDepthAbs.min())
        print(H, W)
        assert H == 480
        assert W == 640
        pred = imgDepthAbs
        #pc.create_point_cloud(pred)
    
        groundTruth = pc.read_pgm(gt_path)

        depthArray = np.asarray(groundTruth)
        depthArray = depthArray.astype('float32')

        param1 = 351.3
        param2 = 1092.5
        depthArray = param1 / (param2 - depthArray)
        depthArray[depthArray > 10.0] = np.NaN
        depthArray[depthArray < 0.0] = np.NaN

        invalid_depths = np.isnan(depthArray)
        valid_depths = np.logical_not(invalid_depths)
        #print("VALID_DEPTHS: ", valid_depths)
        valid_depths = valid_depths.astype("float32")
        #make the NaN values zero
        depthArray[invalid_depths] = 0.0

        #print("groundTruth.shape: ", depthArray.shape)
        #print("pred.shape: ", pred.shape)
        #print("valid_depths.shape: ", valid_depths.shape)

        metric = losses.metric_relative_difference(depthArray, pred, valid_depths)
        print("metric: ", metric)
        total_metric += metric

    average_metric = total_metric/len(dataset)
    print("Metric (avg): ", average_metric)
    
    os._exit(0)

if __name__ == '__main__':
    main()
