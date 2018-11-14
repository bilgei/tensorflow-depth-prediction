import os
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

HEIGHT = 480
WIDTH = 640

def create_point_cloud(depth):
    # Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02

    [xx, yy] = np.meshgrid(range(0, WIDTH), range(0, HEIGHT))
    X = (xx - cx_d) * depth / fx_d
    Y = (yy - cy_d) * depth / fy_d
    Z = depth

    #show
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ## defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    #for c, m, zlow, zhigh in [('r', 'o', -50, -25)]:
    #    ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten(), cmap=plt.cm.coolwarm, marker=m)

    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')            
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

#test
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

