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
#from pyntcloud import PyntCloud


def predict(model_data_path, image_path):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)

    print("img",img.shape)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

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

        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
        fig.savefig('depth.png', bbox_inches='tight')

        return img,pred

def draw_point_cloud(pred):
    # Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02

    imgDepthAbs = pred

    [xx, yy] = np.meshgrid(range(0, W), range(0, H))
    X = (xx - cx_d) * imgDepthAbs / fx_d
    Y = (yy - cy_d) * imgDepthAbs / fy_d
    Z = imgDepthAbs

    numpoints = 1000
    indices = np.arange(0, 480*640)
    indices = npr.choice(indices,size=numpoints,replace=False)
    #points3d = np.concatenate((X,Y,Z), axis=None)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for c, m, zlow, zhigh in [('r', 'o', -50, -25)]:
        #rgbflat = rgb.reshape([640*480,3])
        #print("rgbflat", rgbflat.shape)
        #rgbsampled = rgbflat[indices,:]/255
        #print("dtype", rgbsampled.dtype)
        ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten(), cmap=plt.cm.coolwarm, marker=m)
        #ax.scatter(X.flatten()[indices], Y.flatten()[indices], Z.flatten()[indices], c=rgb.reshape([numpoints,3])[indices], cmap=plt.cm.coolwarm, marker=m)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')            
    #plt.show()

    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).transpose()
    points = [tuple(x) for x in points]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    print("Vertex shape: ", vertex.shape)
    #vertex_color = np.array(Z.flatten(), dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    #n = len(vertex)
    #vertex_all = np.empty(n, vertex.dtype.descr + vertex_color.dtype.descr)

    #for prop in vertex.dtype.names:
    #    vertex_all[prop] = vertex[prop]

    #for prop in vertex_color.dtype.names:
    #    vertex_all[prop] = vertex_color[prop]

    #ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)

    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)

    ply.write('point_cloud.ply')

def read_pgm(filename):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as infile:
        header = infile.readline()
        width, height, maxval = [int(item) for item in header.split()[1:]]
        #print header
        ximg = np.fromfile(infile, dtype=np.uint16).reshape((height, width))
        return ximg

def metric_relative_difference(gt, pred, valid_depths):
    result = 0.0
    sum_relative_difference_total = 0.0
    nr_valid_pixel_total = 0.0

    print("VALID_DEPTHS: ", valid_depths)

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    valid_depths = valid_depths.reshape(-1)
    print("valid_depths: ", valid_depths)

    pred = pred * valid_depths
    target = gt * valid_depths
    difference = pred-target
    processed_difference = difference**2
    relative_difference = np.divide(processed_difference, target + 1e-5)

    sum_relative_difference = np.sum(relative_difference)
    nr_valid_pixel_total = np.sum(valid_depths)

    print("relative_difference: ", sum_relative_difference)
    print("nr_valid_pixels: ", nr_valid_pixel_total)

    result = sum_relative_difference/nr_valid_pixel_total

    return result

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    rgb, pred = predict(args.model_path, args.image_paths)
    imgDepthAbs = pred[0,:,:,:]
    imgDepthAbs = cv2.resize(imgDepthAbs, (640,480))
    rgb = cv2.resize(rgb[0,:,:,:], (640,480))####    
    print(imgDepthAbs.shape)
    [H, W] = imgDepthAbs.shape
    print(imgDepthAbs.max(), imgDepthAbs.min())
    print(H, W)
    assert H == 480
    assert W == 640
    pred = imgDepthAbs
    draw_point_cloud(pred)

    gt_path = 'D:/Bilge/FCRN-DepthPrediction-master/tensorflow/images/d-1315108725.462695-2617891435.pgm'
    groundTruth = read_pgm(gt_path)

    depthArray = np.asarray(groundTruth)
    depthArray = depthArray.astype('float32')

    param1 = 351.3
    param2 = 1092.5
    depthArray = param1 / (param2 - depthArray)
    depthArray[depthArray > 10.0] = np.NaN
    depthArray[depthArray < 0.0] = np.NaN

    invalid_depths = np.isnan(depthArray)
    valid_depths = np.logical_not(invalid_depths)
    print("VALID_DEPTHS: ", valid_depths)
    valid_depths = valid_depths.astype("float32")
    #make the NaN values zero
    depthArray[invalid_depths] = 0.0

    print("groundTruth.shape: ", depthArray.shape)
    print("pred.shape: ", pred.shape)
    print("valid_depths.shape: ", valid_depths.shape)

    metric = metric_relative_difference(depthArray, pred, valid_depths)
    print("metric: ", metric)
    
    os._exit(0)

if __name__ == '__main__':
    main()
