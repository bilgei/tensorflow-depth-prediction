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

def draw_point_cloud(rgb, pred):
    # Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02

    imgDepthAbs = pred[0,:,:,:]
    imgDepthAbs = cv2.resize(imgDepthAbs, (640,480))
    rgb = cv2.resize(rgb[0,:,:,:], (640,480))
    print("rgb size", rgb.shape)
    print(imgDepthAbs.shape)
    [H, W] = imgDepthAbs.shape
    print(imgDepthAbs.max(), imgDepthAbs.min())
    print(H, W)
    assert H == 480
    assert W == 640

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
        #print("rgbflat",rgbflat.shape)
        #rgbsampled = rgbflat[indices,:]/255
        #print("dtype",rgbsampled.dtype)
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

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    rgb,pred = predict(args.model_path, args.image_paths)

    draw_point_cloud(rgb, pred)

    os._exit(0)

if __name__ == '__main__':
    main()
