import cv2 #handles image processing tasks
import numpy as np


#load colorization model
prototxt = r"C:\Users\woner\PycharmProjects\image_colorization\Colorize\Model\colorization_deploy_v2.prototxt" #structure
model = r"C:\Users\woner\PycharmProjects\image_colorization\Colorize\Model\colorization_release_v2.caffemodel" #learned weights(start using model)
points = r"C:\Users\woner\PycharmProjects\image_colorization\Colorize\Model\pts_in_hull.npy" #color data

#Read the model
net = cv2.dnn.readNetFromCaffe(prototxt, model) #dnn->deep neural network
#load color data
pts = np.load(points)

# Prepare model
class8 = net.getLayerId("class8_ab") #this layer predicts the color classes
conv8 = net.getLayerId("conv8_313_rh")#another layer that handles final color predictions
pts = pts.transpose().reshape(2, 313, 1, 1) # 2 color channels and 313 possible values 1 by 1 convolution
net.getLayer(class8).blobs = [pts.astype("float32")]#convert color pts to 32-bit float, insert into class8 layer
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]#help with color prediction accuracy

# Load B&W image
image_path = r"img.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found!")
    exit()

# Preprocess & colorize (unchanged)
scaled = image.astype("float32") / 255.0  #convert to floating point numbers from 0.0-1.0 instead of 0-255
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)#from BGR to LAB
resized = cv2.resize(lab, (224, 224))#size of input

L = cv2.split(resized)[0] - 50 #seperate channels so L=lightness channel, sub 50 to center values

net.setInput(cv2.dnn.blobFromImage(L))#prepare image in format for model
ab = net.forward()[0].transpose((1, 2, 0))#start predicting A and B
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))#resize colors to original image size

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)#combine L and AB
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) #convert from LAB to BGR
colorized = (255 * np.clip(colorized, 0, 1)).astype("uint8")


def resize_with_aspect_ratio(img, width=None, height=None):
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    return cv2.resize(img, dim)

# Set display width 600 pixels
display_width = 600

# Resize both images for display (maintain aspect ratio)
small_original = resize_with_aspect_ratio(image, width=display_width)
small_colorized = resize_with_aspect_ratio(colorized, width=display_width)

# display windows
cv2.imshow("Original", small_original)
cv2.imshow("Colorized", small_colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("colorized_output.jpg", colorized)
print("Colorized image saved!")