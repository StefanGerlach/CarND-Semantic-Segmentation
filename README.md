# Semantic Segmentation
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[image1]: ./runs/1528643478.426405/um_000003.png "Splash"
[image2]: ./runs/1528643478.426405.png "Training"


### Overview

This project uses FUlly Convolutional Neuronal Networks to label every pixel of road images. This is done with the FCN-8 architecture using VGG16-feature extractor ([Paper](https://arxiv.org/abs/1605.06211)). 

![Splash][image1]

### Dataset

For this project, the Kitti Road dataset is used. I downloaded the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip). 

### Pre-Trained Network

To only fine tune the network on my GPU, I used the pretrained VGG16-Network (customized to fully convolutional architecture) from the course-material.

### Training

To train the network, I implemented all necessary function for 
 - skip connections of the respective layers
 - loss and optimization definition
 - hyper parameter learning rate = 1e-4
 - hyper parameter epochs = 60
 
 ![Training][image2]

### Test after training

To test the results, only visual verification is used. Please have a look into the run-directory to see the latest output.

### Refelection

- To further monitor training, tensorboard should be used. Additionally, the Intersection Over Union - metric will help to verfiy the training progress.
- More advanced image augmentation should be used. I only added horizontal flipping.
- More recent architectures like DeepLab or other U-Net-like Encoder-Decoder architectures with skip connections may help to improve the result.
- More labels could be used to increase the amount of possible 'objects/areas' the network can recognize in the scene.
- More data (other datasets) could be used to increase the quality of the model.


### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)


### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
