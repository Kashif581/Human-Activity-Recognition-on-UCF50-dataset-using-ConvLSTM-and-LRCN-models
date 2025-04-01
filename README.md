# Human-Activity-Recognition-on-UCF50-dataset-using-ConvLSTM-and-LRCN-models

# **Convolutional Neural Network (CNN):**
A Convolutional Neural Network (CNN or ConvNet) is a type of deep neural network that is specifically designed to work with image data and excels when it comes to analyzing the images and making predictions on them.

It works with kernels (called filters) that go over the image and generate feature maps (that represent whether a certain feature is present at a location in the image or not) initially it generates few feature maps and as we go deeper in the network the number of feature maps is increased and the size of maps is decreased using pooling operations without losing critical information.

![download (3)](https://github.com/user-attachments/assets/35ba827f-0930-4e49-92af-7c36d740d83a)

Each layer of a ConvNet learns features of increasing complexity which means, for example, the first layer may learn to detect edges and corners, while the last layer may learn to recognize humans in different postures.

# **Long Short-Term Memory (LSTM):**

An LSTM network is specifically designed to work with a data sequence as it takes into consideration all of the previous inputs while generating an output. LSTMs are actually a type of neural network called Recurrent Neural Network, but RNNs are not known to be effective for dealing with the Long term dependencies in the input sequence because of a problem called the Vanishing gradient problem.
LSTMs were developed to overcome the vanishing gradient and so an LSTM cell can remember context for long input sequences.

![download (4)](https://github.com/user-attachments/assets/31258909-68c2-4e27-9ac7-3569e4ebbe0b)

This makes an LSTM more capable of solving problems involving sequential data such as time series prediction, speech recognition, language translation, or music composition. But for now, we will only explore the role of LSTMs in developing better action recognition models.

Now let’s move on towards the approach we will implement in this tutorial to build an Action Recognizer. We will use a Convolution Neural Network (CNN) + Long Short Term Memory (LSTM) Network to perform Action Recognition while utilizing the Spatial-temporal aspect of the videos.

# **CNN + LSTM**

We will be using a CNN to extract spatial features at a given time step in the input sequence (video) and then an LSTM to identify temporal relations between frames.
![download (5)](https://github.com/user-attachments/assets/44047a41-767c-4a11-b203-2cdd0f482a19)
The two architectures that we will be using to use CNN along with LSTM are:


*   ConvLSTM
*   LRCN


Both of these approaches can be used using TensorFlow.

# **Step 1: Download and Visualize the Data with its Labels**

In the first step, we downloaded and visualize the data along with labels to get an idea about what we will be dealing with. We are using the UCF50 – Action Recognition Dataset, consisting of realistic videos taken from YouTube which differentiates this data set from most of the other available action recognition data sets as they are not realistic and are staged by actors. The Dataset contains:


*   50 Action Categories
*   25 Groups of Videos per Action Category
*   133 Average Videos per Action Category
*   199 Average Number of Frames per Video
*   320 Average Frames Width per Video
*   240 Average frame height per Video
*   26 Average Frames Per Seconds per Video

For visualization, we picked 20 random categories from the dataset and a random video from each selected category and visualized the first frame of the selected videos with their associated labels. This way we’ll be able to visualize a subset ( 20 random videos ) of the dataset.CodeText
![download (6)](https://github.com/user-attachments/assets/c3495287-74a0-45a8-9e0e-290700f256d3)

# **Step 2: Preprocess the Dataset**
Next, we perform some preprocessing on the dataset. First, we read the video files from the dataset and resized the frames of the videos to a fixed width and height, to reduce the computations and normalized the data to range [0-1] by dividing the pixel values by 255, which makes convergence faster while training the network.


***Create a Function for Dataset Creation***
We create a function frames_extraction() that will create a list containing the resized and normalized frames of a video whose path is passed to it as an argument. The function will read the video file frame by frame, although not all frames are added to the list as we will only need an evenly distributed sequence length of frames.

# **Step 3: Split the Data into Train and Test Set**
As of now, we have the required features (a NumPy array containing all the extracted frames of the videos) and one_hot_encoded_labels (also a Numpy array containing all class labels in one hot encoded format). So now, we split our data to create training and testing sets. We also shuffle the dataset before the split to avoid any bias and get splits representing the overall distribution of the data.

# **Step 4: Implement the ConvLSTM Approach**
In this step, we implement the first approach by using a combination of ConvLSTM cells. A ConvLSTM cell is a variant of an LSTM network that contains convolutions operations in the network. it is an LSTM with convolution embedded in the architecture, which makes it capable of identifying spatial features of the data while keeping into account the temporal relation.

# **Step 5: Implement the LRCN Approach**
In this step, we implement the LRCN Approach by combining Convolution and LSTM layers in a single model. Another similar approach can be to use a CNN model and LSTM model trained separately. The CNN model can be used to extract spatial features from the frames in the video, and for this purpose, a pre-trained model can be used, that can be fine-tuned for the problem. And the LSTM model can then use the features extracted by CNN, to predict the action being performed in the video.

But here, we implement another approach known as the Long-term Recurrent Convolutional Network (LRCN), which combines CNN and LSTM layers in a single model. The Convolutional layers are used for spatial feature extraction from the frames, and the extracted spatial features are fed to LSTM layer(s) at each time-steps for temporal sequence modeling. This way the network learns spatiotemporal features directly in an end-to-end training, resulting in a robust model.

# **Step 6: Test the Best Performing Model on YouTube videos**
From the results, it seems that the LRCN model performed significantly well for a small number of classes. so in this step, we put the LRCN model to test on some youtube videos.




