# image generation

 Add short description of project here > this project generates numbers in different sizes and shapes

![add image descrition here](direct image link here)

## The Algorithm

This code is for preparing a dataset for training a machine learning model. Specifically, it is for training a model to classify handwritten digits from the MNIST dataset.

The code does the following:

Imports the necessary libraries: tensorflow, keras, numpy, and matplotlib.
Defines some helper functions: show_min_max and plot_image, which are used to display images and their minimum and maximum values.
Loads the MNIST dataset using keras.datasets.mnist.load_data() and stores the training and test images and labels in variables train_images, train_labels, test_images, and test_labels.
Reshapes the images to the correct format for use in a machine learning model, depending on the value of K.image_data_format(), which is a configuration setting in Keras. If K.image_data_format() is set to 'channels_first', the images are reshaped to have a shape of (num_samples, 1, img_rows, img_cols). If K.image_data_format() is set to 'channels_last', the images are reshaped to have a shape of (num_samples, img_rows, img_cols, 1).
Displays an image from the training set using the plot_image function and displays its minimum and maximum values using the show_min_max function.
Converts the image data to float32 and scales it so that the pixel values are between 0 and 1, by dividing the images by 255. This is a common preprocessing step for image data.
Displays another image from the training set and its minimum and maximum values after scaling.
Converts the integer labels to one-hot encoded labels, which are arrays with a 1 in the position corresponding to the label and 0s everywhere else. This is a common representation for categorical data in machine learning.

## Running this project

1. if you would like to run the project without changing anything, press run. if you would like to get a separate output, change the (train_images /= 255) and the 
(test_images /= 255) values to ge a different shape.
2. no required libraries

[View a video explanation here][video link](https://youtu.be/Y6-yngzlN2M)
