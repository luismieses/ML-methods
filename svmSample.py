import pandas as pd 
import numpy as np 
from sklearn import svm

import matplotlib.pyplot as plt 
import seaborn as sns; sns.set(font_scale=1.2)

#Read the files and select the columns to feed the svm algorithm 

genes = pd.read_csv('macrophages.csv')

X = genes.iloc[:,1:9]
Y = genes.iloc[:,9:10]


# build train and test dataset; the shuffle variable its set to False because that way it reads the 
# csv file in order not random

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .35, shuffle = False, random_state = 100)



from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

#Size of the whole graph this is for visualization

fig = plt.figure(figsize=(30,35))

# Function to visualize the svm work

def run_model(model, alg_name, plot_index):

    #build the model on training data

    model.fit(X_train, y_train)

    # make predictions for test data

    y_pred = model.predict(X_test)

    # calculate the accuracy score

    accuracy =  '%.2f'%(accuracy_score(y_test, y_pred) * 100)

    # Compare the prediction result with ground truth

    color_code = {'skin':'red', 'tongue':'blue'}

    #Plot both the testing and the prediction of the graph

    plt.rcParams.update({'font.size' : 30})
    
    colors = [color_code[x] for x in y_test.iloc[:,0]]
    plt.scatter(X_test.iloc[:,0], X_test.iloc[:,5], color=colors, marker='x', label='X = Ground truth')
   
    colors = [color_code[x] for x in y_pred]   
    plt.scatter(X_test.iloc[:,0], X_test.iloc[:,5], color=colors, marker='s', facecolors='none', label='Square = Prediction')

    

    #Plot the legend of the graph 

    plt.legend(loc="lower right")

    # manually set legend color to black

    leg = plt.gca().get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')
    leg.legendHandles[1].set_facecolors('none')
    plt.xlabel('Gene intensity on t-0', fontsize = 20)
    plt.ylabel('Gene intensity on t-120', fontsize = 20)

    plt.title(alg_name + " Accuracy: " + str(accuracy), fontsize = 20)



	

# Call the function svm part

from sklearn.svm import SVC
model = SVC()
run_model(model, "SVM Classifier", 1)


# This is what shows the graph after you run the code 

plt.show()
