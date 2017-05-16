from VLAD.VLAD import *
import argparse
import glob
import cv2


#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--descriptorsPath", required = True,
	help = "Path to the file that contains the descriptors")
ap.add_argument("-w", "--numberOfVisualWords", required = True,
	help = "number of visual words or clusters to be computed")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where the computed visualDictionary will be stored")
args = vars(ap.parse_args())

#args
path = args["descriptorsPath"]
k = int(args["numberOfVisualWords"])
output=args["output"]

#computing the visual dictionary
print("estimating a visual dictionary of size: "+str(k)+ " for descriptors in path:"+path)

with open(path, 'rb') as f:
    descriptors=pickle.load(f)

visualDictionary=kMeansDictionary(descriptors,k)

#output
file=output+".pickle"

with open(file, 'wb') as f:
	pickle.dump(visualDictionary, f)

print("The visual dictionary  is saved in "+file)
