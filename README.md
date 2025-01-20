# Detecting Osteoporosis using Neural Networks

![image](https://github.com/user-attachments/assets/2b577f02-c73f-473c-9668-ad48b045f476)

Osteoporosis is a condition in which bones become more porous and less dense than normal bones. This increases the risk of fractures by weakening and thinning bones. The spine, hips, and wrists are the most susceptible points. In the absence of a fracture, osteoporosis itself has no symptoms, and most affected individuals are not aware of their condition until they break a bone or have a bone density test. Women are more likely than men to suffer from osteoporosis.Ageing is the most common risk factor. Other contributing factors to osteoporosis include smoking, heavy alcohol use, family history of the disease, and chronic diseases of the kidneys, lungs, stomach, or intestines.




In this repository, we use a multilayered neural network in Deep Learning to predict whether a patient has osteoporosis and the likelihood of the prediction being right.
## **This code is divided into five components:**

### **1. Importing Necessary Libraries**
This covers libraries such as **pandas**, **numpy**, **sklearn**, and **TensorFlow**.

## **2. Data preprocessing**
This stage includes the following subtasks:
- **Loading the Data**: Importing the dataset from CSV.
- **Handling Missing Values**: Imputing missing values with median values.
- **Selecting the Target and Features**: Separate the input features (X) from the target variable (y).
- **Data Splitting**: The dataset is divided into training and testing sets (80:20).
- **Data Scaling**: To achieve faster model convergence the feature values are normalized using **StandardScaler**.


**K-Fold Cross-Validation and Model Building**: Using K-Fold from sklearn.model_selection to divide data into folds and shuffling.
- Model Structure: - The input layer consists of 128 neurons.
- The **Hidden Layers** have 64 and 32 neurons with ReLU activation.
- **Batch Normalization** normalizes the layer outputs to reduce covariate shifts.
- **Dropout** is used to remove random neurons during training to prevent overfitting.
- **Output Layer**: A single neuron with a **sigmoid** activation function for binary classification (osteoporosis or not).
- **Early Stopping**: Ends model training when performance no longer increases and accepts the convergence point.
- **Compilation**: Using the **Adam(Adaptive Moment Estimation) optimizer**, the loss function is **binary crossentropy**, and accuracy is the measure.
- **Training**: Run the model for 100 epochs with a batch size of 16 and monitor the validation loss.





## **4. Model evaluation**
In this step, we assess the model's performance on the test set.
- **prediction**: The model outputs a Boolean value, which is then transformed to an integer (0 or 1) for evaluation.
- **metrics**: Accuracy, confusion matrix, and classification report are used to evaluate precision, recall, and F1-score.

### **5. New Patient Predictions**
This section includes capabilities for predicting osteoporosis in a new patient.
- **get_patient()**: This function gathers the patient information while ensuring inputs are in the correct format.
- **predict_new()**: This function uses the learned model to determine whether the patient is at risk of osteoporosis. The function accepts patient data as input and provides a prediction (0 = No Osteoporosis, 1 = Osteoporosis).

This systematic method allows for clear comprehension and efficient development while also ensuring the project's maintainability.


## **Conclusion**
Implementing a neural network-based osteoporosis detection system is a helpful tool for early diagnosis and prevention. Using machine learning approaches, this system can help healthcare practitioners make quick judgments. K-fold cross-validation, batch normalization, dropout, and early termination improve model resilience, generalizability, and reduce overfitting. With more data gathering and fine-tuning, this model has the potential to improve its accuracy and become a more effective medical diagnostic tool. Continuous enhancements, such as adding more patient data and investigating advanced neural network topologies, can help to improve the system's prediction ability.
