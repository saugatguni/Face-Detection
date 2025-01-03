from imp.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2

#we need to construct argument parser and parse the argument
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", type=str, required=True,
                help="path to input image")
ap.add_argument("-m","--model",typr=str,
                default="mmod_human_face_detector.dat",
                help="path to pre-trained dlib's CNN face detector model in device")
ap.add_argument("-u","--upsample", default=1,
                help="number of times to upsample an image")
args=vars(ap.parse_args())

print("(Info) loading Cnn face detector")
detector=dlib.cnn_face_detector_model_v1(args["model"])

#load the input image from device, resize it, and convert it from BGR to RGB which is what dlib expects
image=cv2.imread(args["image"])
image=imutils.resize(image,width=600)
rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#now perform face detection using dlib's face detector
start=time.time()
print("[INFo] Performing face detection")
results=detector(rgb, args["upsample"])
end=time.time()
print("[INFO]Face detection took (:4f) seconds".format(end-start))

#convert the resulting dlib rectangle objects into bounding boxes, then ensure the bounding boxes are all within the bounds of the input image

boxes=[convert_and_trim_bb(image, r.rect) for r in results]

for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) #draw the bounding box on the image

    cv2.imshow("Output", image)
    cv2.waitKey(0)