import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import models
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
    
def displayImage(image, title):
    plt.imshow(image)#, cmap='gray')
    plt.title(title)
    plt.show()

def predict(model_data_path, image_path, net, input_node, prev_rgb, prev_pred):
    img = Image.open(image_path)
    img = img.resize([WIDTH, HEIGHT], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)  

    with tf.Session() as sess:        
        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})

        new_pred = pred[0,:,:,0]
        if prev_pred is None:
            prev_pred = new_pred
        
        prediction = (np.array(new_pred) + np.array(prev_pred)) / 2  #Approach #1
        
#        # Plot result
#        fig = plt.figure()
#        ii = plt.imshow(new_pred, interpolation='nearest')
#        fig.colorbar(ii)
#        plt.title("Prediction")
#        plt.show()
        # #fig.savefig('depth.png', bbox_inches='tight')

        return img, prediction

def main():
    log_path = "logs.txt"
    log_file = open(log_path,'w')    
    
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
    total_abs_relative_difference = 0
    calculated_image_no = 0
    
    results_path = 'results.txt'
    results_file = open(results_path,'w')
    
    for i, sample in enumerate(dataset):
        if i == 0:
            prev_pred = None
            prev_rgb = None

        image_path = sample[0]
        gt_path = sample[1]

        print("image_path: ", image_path)
        print("gt_path: ", gt_path)
        log_file.write("image_path: " + image_path)
        log_file.write("gt_path: " + gt_path)

        # Predict the image        
        rgb, pred = predict(model_path, image_path, net, input_node, prev_rgb, prev_pred)
        prev_pred = pred
        prev_rgb = rgb

#        if i == len(dataset) - 1:
#            fig = plt.figure()
#            ii = plt.imshow(pred, interpolation='nearest')
#            fig.colorbar(ii)
#            plt.show()

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

        invalid_depths = np.isnan(depthArray)
        valid_depths = np.logical_not(invalid_depths)
        #print("VALID_DEPTHS: ", valid_depths)
        valid_depths = valid_depths.astype("float32")
        #make the NaN values zero
        depthArray[invalid_depths] = 0.0

        RMSE_linear = losses.metric_RMSE(depthArray, pred, valid_depths, "linear")
        print("RMSE (linear): ", RMSE_linear)
        log_file.write("RMSE (linear): " + str(RMSE_linear))
        total_RMSE_linear += RMSE_linear

        RMSE_log = losses.metric_RMSE(depthArray, pred, valid_depths, "log")
        print("RMSE (log): ", RMSE_log)
        log_file.write("RMSE (log): " + str(RMSE_log))
        total_RMSE_log += RMSE_log

        abs_relative_difference = losses.relative_difference_metric(depthArray, pred, valid_depths, "abs")
        print("abs_relative_difference: ", abs_relative_difference)
        log_file.write("abs_relative_difference: " + str(abs_relative_difference))
        total_abs_relative_difference += abs_relative_difference

        log_file.write("")
        calculated_image_no += 1

    average_RMSE_linear = total_RMSE_linear / calculated_image_no
    print("---------- RMSE Linear (avg): ", average_RMSE_linear)
    average_RMSE_log = total_RMSE_log / calculated_image_no
    print("---------- RMSE Log (avg): ", average_RMSE_log)
    average_abs_relative_difference = total_abs_relative_difference / calculated_image_no
    print("---------- abs_relative_difference (avg): ", average_abs_relative_difference)
    
    results_file.write("Averageing Results:")
    results_file.write("---------- RMSE Linear (avg): " + str(average_RMSE_linear))
    results_file.write("---------- RMSE Log (avg): " + str(average_RMSE_log))
    results_file.write("---------- abs_relative_difference (avg): " + str(average_abs_relative_difference))
    results_file.close()

    log_file.close()
    
    os._exit(0)

if __name__ == '__main__':
    main()
