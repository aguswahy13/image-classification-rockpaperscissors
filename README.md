# Image Classification Rockpaperscissors Datasets
This program recognizes the shape of the hand that makes up scissors, rock, or paper. Program used artificial neural network (ANN) more than 1 hidden layer with TensorFlow. Rockpaperscissors datasets contain rock, paper, and scissor image in each folder. 

## Detail process followed by
1. Program is used on Google Colab
2. Training data 60% and validation data 40% of the total dataset
3. Implement image augmentation (apply more image augmentation)
4. Using image data generator
5. Use of callbacks
6. Model uses a sequential model 
7. The accuracy of the model is at least 85%.
8. Can predict images uploaded to Google Colab

## Suggestion for the program
1. Transfer learning pre-trained model, learning rate, dropout padding stride
2. Plot for visualization of accuracy and loss and the plot works to find out whether the accuracy/loss results from the model made are overfitting, underfitting or already goodfitting
3. Dropout to overcome overfitting

## Recommendation for the datasets
Add rock, paper, scissor datasets that not in the green background because when using rock, paper, and scissors image that not in the green background sometimes do the wrong prediction result than it should be

## Reference
### Technique
1. Transfer learning pre-trained VGG16 ResNet AlexNet [1](https://pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/) and [2](https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/)
2. [Padding dan stride](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)
3. [Learning rate](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)
4. Plot akurasi dan loss data [1](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/) and [2](https://www.tensorflow.org/tutorials/images/classification#visualize_the_model)
5. [Dropout regularization](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)
6. [BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/)
7. [Callback](https://keras.io/api/callbacks/)
8. Image Preprocessing OpenCV [1](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html) and [2](https://www.programmersought.com/article/48501109629/)

### Other
1. Augmentasi gambar [1](https://www.kaggle.com/gimunu/data-augmentation-with-keras-into-cnn), [2](https://keras.io/api/preprocessing/image/), and [3](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)
2. Split data: [split-folders](https://pypi.org/project/split-folders/), [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), and [validation_split](https://keras.io/api/preprocessing/image/)
3. Underfitting dan Overfitting [1](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/) and [2](https://www.youtube.com/watch?v=u2TjZzNuly8.)
4. [Setting private output mode in Google colab](https://stackoverflow.com/questions/55194081/what-is-private-output-mode-in-google-colab)
