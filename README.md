# Aritfical_neural_net_using_tensorflow

### An application of Artificial Neural Networks using Tensorflow library from Google.  ###

#### Artificial Neural Networks was explored to get better understanding of both popular AI algorithm and the industry standard library used to create them. Algorithm used to perform supervised learning in feed-forward neural architectures is backpropagation and RMSprop. ####

#### Data is from Thermopile Arrays which are passive sensors that register infrared from the environment which is correlated with temperature and can act as a form of heat detector. Data consists of a collection of indoor and outdoor samples collected with human subjects in front of the sensor, with categories such as samples with no subject; with a single subject from 1', 3', and 6'; two subjects from 3'; and three subjects from 3'. ####

#### Backprop algirithm used here uses gradient descent which efficiently computes the gradient passing  through  the  network, one forward and then one backword, propagating the correction using chain rule from calculus. The backpropagation algorithm calculates the gradient of the network’s error, that  is  error produced by each neuron in the network and then finds out how that neuron data should be tweaked in order to decrease that error to converge to a solution. ####

#### RMSprop was used which is an adaptive learning rate method to improve gradient descent leraning performance. RMSprop calculates and keeps a moving average of squared gradients for each weight update. During gradient step, the gradient is divided by square Root of mean square (moving average). Both gradient descent algorithm uses momentum to reach faster convergence without the risk of oscillations.

#### There were two kinds of classifications encountered during this project, Binary and Multi-class classification. Labels were created to help in gradient descent process, labels such as outdoor, indoor, quantity, presence distance were used. Eight mini-datasets were created to classify different questions for both indoors and outdoors separately and one together such as: ####

  ▪ Presence or absence of subject (I'd suggest the 3' dataset for this)
  ▪ Distance of single subject (it's up to you whether you include no-subject in this)
  ▪ Number of subjects (1, 2, 3, or none)
  ▪ Indoor or Outdoor

#### Accuracy metrics were used which provides the percentage of correct predictions compared to actual label. Also, sum-mean square error was used to see error loss during training and to plot on graph. K-fold cross validation method is used to split the data into training and validation data to tune the hyper-parameters in small limited data set to estimate how the model is expected to perform when put to test on unseen data during training of network. It helps us to use more data and get more metrics on the model performance on different data sets and help fine tune the parameters. ####

#### Other tests were performed to check which configuration is better for the learning algorithm by changing the hyperparamers to find statisial significant config. Shapiro  wilk  test  is  performed  on  all  three  test  set  ac-curacies.  This  test,  tests  the normality of the data. Scipy library of python was used to find results of the shapiro test. ANOVA test is performed on the test accuracy sets on three different parameters which provides us the evidence about the differences among means. Stats model library was used to find results of ANOVA test. t-test is performed on accuracy test set to determine if there is significant difference between the  means among all three tests. Scipy library was used to find results of t-test. In general, if t-test rejects hypothesis, the higher accuracy test parameter withless standard deviation is the best candidate. ####

#### Sample comparison for Backpropagation using Gradient descent with 1 hidden layer, 20 neurons in hidden layer, relu activation function and 20 epochs, while only changing values of learning rate and momentum as 0.09/0.4/0.95 learning rate and 0.4/0.7/0.9 momentum. ####

### NOTE: https://colab.research.google.com/ was used for initial ingestion, exploration and validation of results. Pycharm IDE was used to get full results later. ###


All three tests with 7 different mini-datasets produces following results:

<img src="https://user-images.githubusercontent.com/54603828/124204897-051ec780-daae-11eb-97f6-2b3f4b08025c.png" width="600" height="300"> 

And when plotted on a box plot, initial observations prooves that test one and two are statisitcally better than the thrid while ANOVA and t-tests confirm the hypothesis :

<img src="https://user-images.githubusercontent.com/54603828/124205036-65ae0480-daae-11eb-8159-6b86ac725ed0.PNG" width="400" height="300"> 

Binary_class_predictions_compared_with_true_values with example:

<img src="https://user-images.githubusercontent.com/54603828/124195136-cb42c680-da97-11eb-9ae2-f659ec367f04.PNG" width="750" height="350"> 

dist_prediction_rmsprop_comparison_with_true_values:

<img src="https://user-images.githubusercontent.com/54603828/124195269-178e0680-da98-11eb-8f59-e1a6c202fbcb.PNG" width="750" height="350"> 

rmsprop_accuracy_loss_over_epochs:

<img src="https://user-images.githubusercontent.com/54603828/124195044-9afb2800-da97-11eb-9c53-a911b91897f9.PNG" width="350" height="350">

False_Negatives-False_Positives_over_epochs:

<img src="https://user-images.githubusercontent.com/54603828/124195325-38eef280-da98-11eb-90b4-51b5c6d56f69.PNG" width="350" height="350">

Accuracy_loss_over_epochs:

<img src="https://user-images.githubusercontent.com/54603828/124195368-5328d080-da98-11eb-875e-ec30455041f3.PNG" width="350" height="350">
