#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image



# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()



########################################################################################################################
#               functions
########################################################################################################################


def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    #gender will contains 1 or 0 indecating male or female
    gender =[]
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        gender_current=int(os.path.split(image_path)[1].split(".")[1].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            gender.append(gender_current)
            
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    print("lables")
    print(labels)
    print("gender_current")
    print(gender)
    return images, labels, gender

                                              #############################################################################
def image_prediction(image_path):   #comparing the image in image_path to the data base
    #while 1:
    #if image_path == "2":
    # break
##for image_path in image_paths:
    #print(image_path)
    counter_above=0
    counter_correct=0
    
    found_flag=0
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)

    print(faces)

    for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            #current_gender=int(os.path.split(image_path)[1].split(".")[1].replace("subject", ""))
         #   print(conf)
            
        
            if nbr_actual == nbr_predicted:
                print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
                cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
                cv2.waitKey(1000)
                found_flag=1;
                counter_correct=counter_correct+1
                
                if conf >= 50:
                    counter_above=counter_above+1
                break
            else:
                print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
        
    if found_flag == 0:
        print('new image')
        
    return nbr_predicted

#############################################################################################################################################

# Path to the Yale Dataset
path = './yalefaces'

# Call the get_images_and_labels function and get the face images and the 
# corresponding labels

images, labels, gender = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

image_path = input("Please enter the image path ")                 #####just temporary#######
nbr_predicted =image_prediction(image_path)

print("gender number is %d", nbr_predicted)
counter=0

#print('labeless abara ')
#print(labels)

for f in labels:
 #  print("gowa el f")
 #  print"f %d" %(f)
 # print"%s==%d"% ("counter==",counter)
 #  print"%s==%d"% ("gender==",gender[counter])
   if f==nbr_predicted :
        current_gender=gender[counter]
        break
   counter=counter+1 
#print('barraaaa')
#gender_print="%s==%d"% ("gender==",gender[counter])

if current_gender == 1:
    print('male')
if current_gender == 0:
    print('female')
    



# Append the images with the extension .sad into image_paths
##image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
#image_path = "./predict\subject88.sad"

