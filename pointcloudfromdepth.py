# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 13:37:18 2018

@author: bilgei
"""
import os
import numpy as np
from PIL import Image
#import cv2
from plyfile import PlyData, PlyElement

HEIGHT = 480
WIDTH = 640

def create_point_cloud(depth):    
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02
    
    #imgDepthAbs = cv2.resize(depth, (640,480))
    #print("imgDepthAbs.shape:", imgDepthAbs.shape)
    #[H, W] = imgDepthAbs.shape
    #print(imgDepthAbs.max(), imgDepthAbs.min())
    #print(H, W)
    #assert H == 480
    #assert W == 640

    [xx, yy] = np.meshgrid(range(0, WIDTH), range(0, HEIGHT))
    X = (xx - cx_d) * depth / fx_d
    Y = (yy - cy_d) * depth / fy_d
    Z = depth
    
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).transpose()
    points = [tuple(x) for x in points]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    print("Vertex shape: ", vertex.shape)
    
    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    ply.write('GT_pointcloud.ply')               

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
    
def main():
    image_path = 'D:/Bilge/FCRN-DepthPrediction-master/tensorflow/images/d-1315108725.462695-2617891435.pgm'

    depth = read_pgm(image_path)
    #need to process the depth image, so make it numpy array
    depthArray = np.asarray(depth)
    #change data type to float
    depthArray = depthArray.astype('float32')
    #values bigger tahn this are problematic
    param1 = 351.3
    param2 = 1092.5
    #make the depth values absolute
    depthArray = param1 / (param2 - depthArray)
    #make the problematic values nan so that they dont get involved in the resizing
    depthArray[depthArray>10.0] = np.NaN
    depthArray[depthArray<0.0] = np.NaN
    #prepare the sample and return
    
    print(type(depthArray))
    print("img.shape:",depthArray.shape)
    create_point_cloud(depthArray)
    
    os._exit(0)
    
if __name__ == '__main__':
    main()

