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

    print("img_shape after: ", img.shape)

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
        
        # print("pred1: ", pred[:,0,0,0])
        # print("pred4: ", pred[0,0,0,:])

        new_pred = pred[0,:,:,0]
        if prev_pred is None:
            prev_pred = new_pred

        prediction = (np.array(new_pred) + np.array(prev_pred)) / 2
        print("prediction shape: ", prediction.shape)

        # # Plot result
        # fig = plt.figure()
        # ii = plt.imshow(prediction, interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()
        # #fig.savefig('depth.png', bbox_inches='tight')

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

    total_RMSE_linear = 0
    total_RMSE_log = 0
    total_squared_relative_difference = 0
    total_abs_relative_difference = 0
    calculated_image_no = 0
    
    for i, sample in enumerate(dataset):
        if i == 0:
            prev_pred = None

        image_path = sample[0]
        gt_path = sample[1]

        print("image_path: ", image_path)
        print("gt_path: ", gt_path)

        # Predict the image        
        rgb, pred = predict(model_path, image_path, net, input_node, prev_pred)
        prev_pred = pred

        if i == len(dataset) - 1:
            fig = plt.figure()
            ii = plt.imshow(pred, interpolation='nearest')
            fig.colorbar(ii)
            plt.show()

        H = 480
        W = 640
        pred = cv2.resize(pred, (W, H))
        #pc.create_point_cloud(pred)

        groundTruth = pc.read_pgm(gt_path)

        depthArray = np.asarray(groundTruth)
        depthArray = depthArray.astype('float32')

        param1 = 351.3
        param2 = 1092.5
        depthArray = param1 / (param2 - depthArray)
        depthArray[depthArray > 10.0] = np.NaN
        depthArray[depthArray < 0.0] = np.NaN

        # fig = plt.figure()
        # ii = plt.imshow(depthArray, interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()

        invalid_depths = np.isnan(depthArray)
        valid_depths = np.logical_not(invalid_depths)
        #print("VALID_DEPTHS: ", valid_depths)
        valid_depths = valid_depths.astype("float32")
        #make the NaN values zero
        depthArray[invalid_depths] = 0.0

        # fig = plt.figure()
        # ii = plt.imshow(depthArray, interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()

        RMSE_linear = losses.metric_RMSE(depthArray, pred, valid_depths, "linear")
        print("RMSE (linear): ", RMSE_linear)
        total_RMSE_linear += RMSE_linear

        RMSE_log = losses.metric_RMSE(depthArray, pred, valid_depths, "log")
        print("RMSE (log): ", RMSE_log)
        total_RMSE_log += RMSE_log

        squared_relative_difference = losses.relative_difference_metric(depthArray, pred, valid_depths, "squared")
        print("squared_relative_difference: ", squared_relative_difference)
        total_squared_relative_difference += squared_relative_difference

        abs_relative_difference = losses.relative_difference_metric(depthArray, pred, valid_depths, "abs")
        print("abs_relative_difference: ", abs_relative_difference)
        total_abs_relative_difference += abs_relative_difference

        calculated_image_no += 1

    average_RMSE_linear = total_RMSE_linear / calculated_image_no
    print("---------- RMSE Linear (avg): ", average_RMSE_linear)
    average_RMSE_log = total_RMSE_log / calculated_image_no
    print("---------- RMSE Log (avg): ", average_RMSE_log)
    average_squared_relative_difference = total_squared_relative_difference / calculated_image_no
    print("---------- squared_relative_difference (avg): ", average_squared_relative_difference)
    average_abs_relative_difference = total_abs_relative_difference / calculated_image_no
    print("---------- abs_relative_difference (avg): ", average_abs_relative_difference)

    
    os._exit(0)

if __name__ == '__main__':
    main()
