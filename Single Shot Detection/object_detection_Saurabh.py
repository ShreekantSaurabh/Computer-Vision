# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable    #converts torch tensor to torch variable which contains both tensor and gradient
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap     #data is a folder name. BaseTransform will transform the image which is compatible for neural network. VOC_CLASSES is a dictionary which encodes the classes.
from ssd import build_ssd     #build_ssd is the constructer of SSD neural network
import imageio         #library to process images of a video

# Defining a function that will do the detections inside each image/frame of a video
def detect(frame, net, transform): # We define a detect function that will take as inputs, an original frame (not gray frame), a ssd neural network, and a transformation to be applied on the images, and that will return the same frame with the detector rectangle.
    height, width = frame.shape[:2] # We get the height and the width of the frame.
    frame_t = transform(frame)[0] # We apply the transformation to our frame.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the numpy array transformed frame into a torch tensor. Permutation/order of the color is changed from its default order 0=red, 1=blue, 2=green
    x = Variable(x.unsqueeze(0)) # We add a fake dimension corresponding to the batch beause neural network doesn't accept single input vector, it has to be converted to batch.
    y = net(x) # We feed the neural network ssd with the image and we get the output y.
    detections = y.data # We create the detections tensor contained in the output y.
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height]. First (width, height) is the coordinate of top left corner of detector rectangle and second is for bottom right corner.
    
    #detections = [batch, no. of classes ie. detectable objects, no. of occurences of objects, (score, x0,y0, x1,y1)]
    for i in range(detections.size(1)): # For every class (detectable objects):
        j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
        #detections[batch, no. of classes/objects, no. of occurences, (score, x0,y0, x1,y1)]  where (x0,y0) and (x1,y1) are the coordinates of top-left and bottom-right corner of detector rectangle.
        while detections[0, i, j, 0] >= 0.6: # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
            pt = (detections[0, i, j, 1:] * scale).numpy() # 1: of tuple (score, x0,y0, x1,y1) gives x0,y0, x1,y1 the coordinates of the points at the upper left and the lower right of the detector rectangle. 
                                                            #Scale will normalise the coordinates between 0 & 1. Conversion of torch tensor to an numpy array is needed because cv2 accepts an array.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object. (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])) are the coordinates of upper-left and lower-right point. (255, 0, 0) is for red color. 2 is for thickness of text to display.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class/object right above the rectangle. eg. dog, human etc. both 2 are size & thickness of text. FONT_HERSHEY_SIMPLEX is the font. (255, 255, 255) is color. LINE_AA for continuous line.
            j += 1 # We increment j to get to the next occurrence.
    return frame # We return the original frame with the detector rectangle and the label around the detected object.

#if __name__ == '__main__':
    
# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd for test phase.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.
#net.size is size of image to be fed to neural network. (104/256.0, 117/256.0, 123/256.0) tuple for scaling color values on which pre-trained model is trained.

# Doing some Object Detection on a video
reader = imageio.get_reader('/home/saurabh/Desktop/Module_2/Code_for_Mac_and_Linux/Code for Mac and Linux/funny_dog.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('/home/saurabh/Desktop/Module_2/Code_for_Mac_and_Linux/Code for Mac and Linux/output.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.