# Osteoporosis

Osteoporosis is a disease in which bones become more porous and less dense than healthy bones. This makes bones thin and weak, increasing the risk of fractures. The most vulnerable areas are the wrists, hips, and spine. Osteoporosis itself, in the absence of a fracture, causes no symptoms, and most people with the disease are unaware of their condition until they break a bone or take a bone density test. Osteoporosis affects more women than men. Ageing is the most common risk factor. Other contributing factors to osteoporosis include smoking, heavy alcohol use, family history of the disease, and chronic diseases of the kidneys, lungs, stomach, or intestines.

![image](https://github.com/user-attachments/assets/2b577f02-c73f-473c-9668-ad48b045f476)

In this Repository, we implement a Simple Neural Network in which we predict whether the patient is suffering from Osteoporosis, along with the chance of the prediction being True.

# This code is categorised into five parts, i.e.
1. **Importing the necessary Libraries**:
       This includes libraries like pandas, numpy, sklearn, TensorFlow and their subsidiaries.
2. **Data Preprocessing**: This involves:<br/>
    a. Loading the data<br/>
    b. handling the missing values<br/>
    c. encoding the features<br/>
    d. Selecting the Target and the Features<br/>
    e. Splitting the data for training and testing<br/>
    f. Scaling the data<br/>

 3. **K-Fold and Model Compilation**:<br/>
    a. We use the Kfold from the sklearn.model_selection library to split the data into 3-folds where we shuffle the data before splitting using the "Shuffle=TRUE" parameter<br/>
    b. We build the basic structure of our neural network that consists of 4 layers, i.e. an Input layer(128 neurons),2 Hidden Layers(64 and 32 neurons respectively) and the output layer(1 neuron), along with the implementation of L2 regularization, Batch Normalization, and the Dropout() to prevent overfitting.<br/>
    c. We use the EarlyStopping() from tensorflow.keras.callbacks library to stop the model from processing the data as soon as the converging point is attained or there is no change in the accuracy of the model for any number of epochs.<br/>
    d. Now we compile our model using the Adam(Adaptive Moment Estimation) optimiser and the Binary_crossentropy as the loss function while considering the accuracy of the model as the metric.<br/>
    e. We now train our model for 100 epochs, keeping the batch size as 16.<br/>

4. **Evaluation**: This is the stage where we get to know if our hard work paid off<br/>
    a. The model predicts a Boolean output, which is in turn converted into a numerical value using the .astype() method with "int32" as a parameter.<br/>
    b. We attain the performance of our model using metrics like Accuracy, Confusion Matrix and Classification Report.<br/>

5. **New Patient**: In this part, we define two functions:<br/>
    a. get_patient(): To get the details of the patient using the Try-Except technique to ensure that the data is in the possible range to be processed.<br/>
    b. predict_new(): To predict whether the patient suffers from Osteoporosis.<br/>
   
