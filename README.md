# gender_model
A very basic machine learning algorithm that uses KNN which predicts gender based on two arguments - height and weight. The model performs around 87% accuracy. 

To successfully run this, you will need to install several dependencies - this command should take care of it:

`pip install numpy scipy matplotlib ipython scikit-learn pandas`

From here, the logic is simple. You generate a dataset based on averages and standard deviations [the averages were taken from sources, the standard deviations were estimated]. Upon generation, you break the data up into test and training data. From here, you use the KNN algorithm [kth nearest neighbor - a simple algorithm that looks at neighbors of a data point and uses a majority vote to classify new data points]. Once this is done, you can check the models accuracy. It tends to hover around 87%.

This is my first model of many to come - a very cool learning experience. 
