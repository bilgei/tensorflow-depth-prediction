import os
import numpy as np

#Alican's method
#format of the output: [ ['pathtorgbimage1','pathtocorrespondingdepthimage1'], ['pathtorgbimage2','pathtocorrespondingdepthimage2'], ... ]
def matchInputOutputPaths(path):
    #print('path',path)
    #open the txt file contains the names of the files. get the names. there are index files with lower case and upper case, so we have to check it first.
    indexPathUpper = path + '/INDEX.txt'
    indexPathLower = path + '/index.txt'

    #print('indexPathLower',indexPathLower)

    if os.path.isfile(indexPathUpper):
        with open(indexPathUpper) as f:
            indexes = f.readlines()
            f.close()
    elif os.path.isfile(indexPathLower):
        with open(indexPathLower) as f:
            indexes = f.readlines()
            f.close()
    else:
        return 'True'
    #return a string if the txt file is problematic
    if len(indexes) < 15:
        return 'True'
    indexes.pop()#there are some problems with the last lines on the index files. this takes care of it.
    indexes = [x.strip() for x in indexes]

    #create two list contains the names of the depth and rgb files.
    numDepth = 0
    namesDepth = []
    numRgb = 0
    namesRgb = []
    for i in range(0, len(indexes)):
        if indexes[i][0] == 'r':
            namesRgb.append(indexes[i])
            numRgb += 1
        elif indexes[i][0] == 'd':
            namesDepth.append(indexes[i])
            numDepth +=1


    #pointer that shows our current point in rgb
    rgbPointer = 0

    #INDEX.txt looks sorted but i did sort the lists of names anyway. we need them sorted in order to match them by timestamp
    namesRgb.sort()
    namesDepth.sort()

    #the output of the function. list of corresponding pairs' addresses
    matchedInputOutput = [['' for x in range(0,2)] for y in range(0, numDepth)]
    #for every depth image, we are looking for closest rgb image in time
    for depthPointer in range(0, numDepth):
        #get the timestamp part frmo the file names
        timeDepth = namesDepth[depthPointer].split('-')
        timeRgb = namesRgb[rgbPointer].split('-')

        timeDepth = float(timeDepth[1])
        timeRgb = float(timeRgb[1])
        #calculate the time differences
        timeDiff = np.absolute(timeDepth - timeRgb)
        #we are picking up where we left off in the previous iteration. this is legit since we sorted both lists
        while rgbPointer<numRgb-1:
            #looking at the next rgb file
            nextTimeRgb = namesRgb[rgbPointer+1].split('-')
            nextTimeRgb = float(nextTimeRgb[1])
            #calculating the time differences
            newTimeDiff = np.absolute(timeDepth - nextTimeRgb)
            #if it gets worse, we gonna stop, otherwise next one is a better match, so update the time differences and increase the rgb pointer
            if newTimeDiff>timeDiff:
                break

            timeDiff = newTimeDiff

            rgbPointer += 1
        #now we found the best match for the depth image. save them in the list with the path
        matchedInputOutput[depthPointer][0] = path + '/' + namesRgb[rgbPointer]
        matchedInputOutput[depthPointer][1] = path + '/' + namesDepth[depthPointer]

    #return the list of rgb image path and depth image path pairs
    return matchedInputOutput