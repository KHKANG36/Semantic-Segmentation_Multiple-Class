# Semantic Segmentation (Multiclass Project)
In this project, I implemented to classify multiple objects on the road pixel by pixel based on the semantic segmentation technique. All codes are written on the iPython notebook. I trained the network with "Cityscape-dataset", and appreciate for using it.  

## Requirement 
- Python 3.0 >
- Tensorflow 1.0 >
- Numpy
- Scipy
- [Cityscape Dataset](https://www.cityscapes-dataset.com/downloads/) 
  
## Run the Project 
Download the Cityscape dataset (You have to sign up). Extract the dataset in the 'data' folder. Make 'training', 'testing' folder and put all the revelant data to each folder. Open the 'Semantic_segmentation_cityscape.ipynb' file and run it sequencially. 

## Project Implementation
1) Neural Network Architecture  
- I used FCN-8 encoder/decoder architecture. I loaded pretrained VGG16 model into tensorflow followed by 1 by 1 convolution for spartial information. Then, I created the layers for a FCN (Fully Convolutional Network) using deconvolution and skip connection technique. For the detailed undertanding, please refer to below architecture image which I used in this project. 
![Test image](https://github.com/KHKANG36/Semantic-Segmentation/blob/master/FCN%20for%20Semantic%20Seg.gif)

2) Class
- I classified 4 objects in this simulation. (Road - Purple, Vehicle - Blue, Ped - Red, Traffic Light - Yellow)
 
## The Result
1) Cityscape dataset test images
- Mostly, it could pretty much classify most of the objects.
![Test image](https://github.com/KHKANG36/Semantic-Segmentation_Multiple-Class/blob/master/runs/berlin_000000_000019_leftImg8bit.png)
![Test image](https://github.com/KHKANG36/Semantic-Segmentation_Multiple-Class/blob/master/runs/berlin_000037_000019_leftImg8bit.png)
![Test image](https://github.com/KHKANG36/Semantic-Segmentation_Multiple-Class/blob/master/runs/berlin_000061_000019_leftImg8bit.png)

2) Recored road video (Germany)
- I applied the trained model to the real road video. It showed good performance. You can see the full video (semantic_output.mp4) in my repo. 
