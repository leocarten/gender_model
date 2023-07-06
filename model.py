import numpy as np
from sklearn.model_selection import train_test_split
import random
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# The average height of women in US is 64 inches
# The average weight of women in US is 170.6 pounds

# The average height of men in US is 69 inches
# The average weight of men in US is 197.9 pounds

def getTargetData(dictionary):
    features = []
    for entry in dictionary.values():
        features.append([entry["gender"]])

    features_array = np.array(features)
    return features_array

def convertToNumpyArray(dictionary):
    features = []
    for entry in dictionary.values():
        features.append([entry["height"], entry["weight"]])

    features_array = np.array(features)
    return features_array

def generateData():
    jsonData = {}

    # below are statistics pulled from online - they are US averages
    # the standard deviation is estimated - tune per your needs
    women_average_height = 64.5
    women_height_std_dev = 2.5
    women_average_weight = 170.6
    women_weight_std_dev = 20.0


    men_average_height = 70
    men_height_std_dev = 3
    men_average_weight = 197.9
    men_weight_std_dev = 25.0

    amountOfData = 1000

    # Generate random data for height and weight following normal distributions
    women_height_data = np.random.normal(women_average_height, women_height_std_dev, amountOfData)
    women_weight_data = np.random.normal(women_average_weight, women_weight_std_dev, amountOfData)

    men_height_data = np.random.normal(men_average_height, men_height_std_dev, amountOfData)
    men_weight_data = np.random.normal(men_average_weight, men_weight_std_dev, amountOfData)

    # Populate the dictionary with height and weight data
    for i in range((amountOfData * 2)):
        if i < amountOfData:
            jsonData[i+1] = {
                "height": women_height_data[i],
                "weight": women_weight_data[i],
                "gender": "Female"
            }
        else:
            jsonData[i+1] = {
                "height": men_height_data[i-amountOfData],
                "weight": men_weight_data[i-amountOfData],
                "gender": "Male"
            }

    # Return the created data
    return jsonData

def shuffleData(data):
    keys = list(data.keys())
    random.shuffle(keys)

    # Create a new dictionary with the shuffled order
    shuffled_dict = {key: data[key] for key in keys}

    # return the newly shuffled dictionary
    return shuffled_dict

def visualize_data(data):
    # visualize the created data
    heights = [entry['height'] for entry in data.values()]
    weights = [entry['weight'] for entry in data.values()]
    genders = [entry['gender'] for entry in data.values()]

    color_mapping = {'Female': 'pink', 'Male': 'blue'}

    for height, weight, gender in zip(heights, weights, genders):
        plt.scatter(height, weight, color=color_mapping[gender])

    # Plot the data
    plt.xlabel('Height [inches]')
    plt.ylabel('Weight [lbs]')
    plt.title('Raw Data Vizualizer')

    plt.grid(True, linestyle='--', alpha=0.5)

    # Add a legend for the gender colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=8, label='Female'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Male')
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

def main():
    # generate the data set
    raw_data = generateData()

    # shuffle the data set
    raw_shuffled_data = shuffleData(raw_data)
    # visualize_data(raw_shuffled_data)

    # convert the data set to a numpy array
    training_data = convertToNumpyArray(raw_shuffled_data)

    # get corresponding target data
    target = getTargetData(raw_shuffled_data)

    # split up the data using the 80/20 rule
    X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.2, random_state=42)

    y_train = np.ravel(y_train)

    # using the built in knn algorithm, we assign the amount of neighbors to 7 using a majority vote
    knn = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    knn.fit(X_train, y_train)

    score = knn.score(X_test, y_test) 
    score_formatted = "{:.2f}".format(score)  
    score_cleaned = score_formatted.replace(".", "") 
    score_int = int(score_cleaned)  
    print("This model performed with ",score_int,"percent accuracy")

main()
