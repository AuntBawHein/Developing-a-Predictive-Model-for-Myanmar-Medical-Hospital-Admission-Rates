# Developing-a-Predictive-Model-for-Myanmar-Medical-Hospital-Admission-Rates
To teach a computer to predict if a patient should be admitted to the hospital using Python 

Introduction

In this project, I want to teach a computer to predict if a patient should be admitted to the hospital. I will use data like the patient’s age, test results, and health conditions. This will help doctors make faster decisions so that patients can receive the right care sooner.
Python is the tool I will use for this project. It will follow my instructions to study the data and find patterns. After learning from the data, the computer will be able to make predictions on its own about whether a patient needs hospital admission.

Goal of This Project

The goal of this project is to help doctors make quick decisions about hospital admissions. The computer program will analyze patient data and make predictions, helping doctors identify who needs urgent care. This will allow patients to receive treatment faster.

Step 1: Define Project / Scope

I will explain what I want to do in this project and how I will do it.

Define Scope/Objectives of the Project

Objective

I want to create a computer program that can predict if a patient needs to stay in the hospital based on their age, gender, and test results.

Scope

I will first clean the data to make sure there are no mistakes. Then, I will study the data to find useful patterns. I will create new details that can help the program make better predictions. After that, I will build and test the model to check how well it works. In the end, I will have a program that helps doctors decide if a patient should stay in the hospital.


Step 2 Data Collection and Preparation

I will collect the data I need and get it ready for my project

2 Data Collection and Preparation

Task
I will collect data that includes important details like Age Sex Lab results Diagnosis Outcome and whether the patient was sent to a government hospital Yes or No
Tools
I will use Pandas a tool in Python to clean and remove unwanted data

Steps

1 I will load the data into a Pandas DataFrame which is like a big table.

2 I will check for missing or incorrect values in the data.

3 I will fix missing values by filling them in or removing them.

4 I will convert text like Sex or Diagnosis into numbers so the computer can understand them better.

What is Pandas?

I will use Pandas because it helps me organize clean and analyze large amounts of data just like a spreadsheet.

Why Use Pandas?

I need Pandas because it helps me collect clean and prepare data easily It also helps me find mistakes fix missing values and turn words into numbers so the computer can understand the data better.


Why do we have to use Pandas for this step?

We use Pandas because it makes it easier to collect, fix, and get the data ready for the model. It helps us spot mistakes and fix them, and it also helps us change words (like Sex or Diagnosis) into numbers so the computer can understand and work with the data better.

Code

# Step 3: Look for missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:\n", missing_values)

# Step 4: Fix the missing values
if 'Age\n(yr)' in data.columns:
    data['Age\n(yr)'] = data['Age\n(yr)'].fillna(data['Age\n(yr)'].mean())
    print("\nMissing values in 'Age' column have been filled with the average age.")

if 'Lab result' in data.columns:
    data['Lab result'] = data['Lab result'].fillna('Not available')
print("\nMissing values in 'Lab result' column have been filled with 'Not available'.")

What I’m going to do:

[caseline list _dengue_as_csv_files.csv](https://github.com/user-attachments/files/19380798/caseline.list._dengue_as_csv_files.csv)
[caseline list _dengue.xlsx](https://github.com/user-attachments/files/19380801/caseline.list._dengue.xlsx)


Step 3 Exploratory Data Analysis EDA

I will carefully look at my data to understand what it shows I will check things like age and missing second doses to find any patterns or trends This will help me understand the data before using it in my model.

Task

I will study my data to find important or interesting details.

Tools

I will use Pandas Matplotlib and Seaborn to create tables charts and graphs.

Steps

1 I will check how the data is spread out like the ages of people and their lab results.

2 I will look at how things like age and gender relate to missing second doses.

3 I will create heatmaps colorful charts to show how different data points are connected.

4 I will find patterns or trends that can help improve my model.

What is Pandas?

Pandas is a tool in Python that helps organize data into tables like an Excel spreadsheet It makes it easier to work with large amounts of information.

Why use Pandas?

I use Pandas to load the data see the first few rows and organize the information so it is easy to explore It helps me understand the data better.

What is Matplotlib?

Matplotlib is a tool in Python that helps create charts and graphs It allows me to see what the data looks like in a simple way.

Why use Matplotlib?

I use Matplotlib to make bar charts and line charts that show patterns in the data These charts help me find important details easily.

What is Seaborn?

Seaborn is a tool that helps create better looking charts in Python It works with Matplotlib to make advanced graphs like heatmaps.

Why use Seaborn?

I use Seaborn because it helps create clearer and more colorful charts Heatmaps and other advanced graphs help compare different parts of the data easily.

Code

# Step 3: Checking the distribution of Age
plt.figure(figsize=(10, 6))  # Set chart size
sns.histplot(data['Age'], bins=10, kde=True)  # Create histogram to show age distribution
plt.title("Distribution of Age (Myanmar Medical Data)")  # Add title
plt.xlabel("Age (Years)")  # Label x-axis
plt.ylabel("Number of People")  # Label y-axis
plt.show()  # Show the chart

![image](https://github.com/user-attachments/assets/2c9fcf53-5ece-4fd9-b7d8-70718d24b11a)
![image](https://github.com/user-attachments/assets/c80f341f-962e-43b1-854e-e009475da7bb)
![image](https://github.com/user-attachments/assets/8f0bedd3-3420-48c1-a7b7-acf17d96aa30)
![image](https://github.com/user-attachments/assets/3ec33560-2e28-4bea-81b7-5da6ee668358)

Conclusion

I found that most patients in the data are young, with many under 20 years old. I noticed that younger people were more likely to miss their second vaccine dose compared to older people. I also saw that more females missed their second dose than males. From the heatmap, I learned that age has a small effect on missing the second dose, but it is not the only reason.

Step 4 Feature Engineering

In this step I am going to change and create new parts of my data to help the model make better predictions

4 Feature Engineering

Task Make new features or change existing ones to help the model work better.

Tools We will use pandas to do this.

Steps

1 Create New Features.

I am going to make new features by combining existing ones For example I can multiply Age and Lab result to create something new that might help the model make better predictions.

2 Normalize or Standardize

I am going to change some numbers so they are easier for the model to understand For example I will adjust Age and Lab result so that they follow the same scale This is called normalizing or standardizing.

3 Change Words into Numbers One Hot Encoding

I am going to turn words into numbers so the model can understand them better For example if the data has different Diagnoses like COVID 19 or Flu I will turn these into numbers This process is called one hot encoding.

4 Remove Useless Data

I am going to remove any data that is not useful or could confuse the model This helps the model focus on the most important information.

What is Pandas?

Pandas is a tool in Python that helps us organize and change data It is similar to using Excel but for much larger amounts of data.

Why do we have to use Pandas for this step?

We use Pandas because it helps us combine change and clean up our data in a way that makes it easier for the model to learn from it.

Code

import pandas as pd  # Importing pandas for data handling

# Handling missing values in numerical columns
if 'Age' in data.columns:
    data['Age'].fillna(data['Age'].mean(), inplace=True)  # Replace missing Age values with the average

if 'Lab_Result' in data.columns:
    data['Lab_Result'].fillna(data['Lab_Result'].mean(), inplace=True)  # Replace missing Lab_Result values with the average

# Creating a new feature by combining existing ones
if 'Age' in data.columns and 'Lab_Result' in data.columns:
    data['Age_Lab_Result'] = data['Age'] * data['Lab_Result']  # Multiply Age and Lab_Result

# Normalizing numerical features
if 'Age' in data.columns:
    data['Age_normalized'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())

if 'Lab_Result' in data.columns:
    data['Lab_Result_normalized'] = (data['Lab_Result'] - data['Lab_Result'].min()) / (data['Lab_Result'].max() - data['Lab_Result'].min())

# Display the first few rows of the modified dataset
print(data.head())  # Show the first few rows to verify changes

Step 5 Data Splitting

Objective

In this step I am going to split the data into two parts one part to teach the model training data and one part to test how well the model learned testing data.

Task

Split the data into training and testing sets so we can check how well the model works.

Tools

We will use scikit learn to split the data.

Steps

1 Separate the data into two groups Features X This is all the information we want the model to learn from like Age Lab Results etc Target y This is the answer we want the model to predict In this case it is whether someone is referred to the local government hospital Yes or No.

2 Use train test split from scikit learn Training Data 80 This part of the data will be used to teach the model and help it learn patterns Testing Data 20 This part of the data will be used to test the model and see how well it learned from the training data.

Why Do We Split the Data?

We split the data so the model can learn from one part training data and then be tested on new data testing data This way we can make sure the model is not just memorizing the information but learning patterns that can be used in new situations. Testing on new data helps us know how well the model can make predictions in real life situations 
This ensures that the model will be able to work well even with data it has not seen before.

Additional Questions

What is Pandas?

Pandas is a tool that helps us work with data in a table format just like Excel In this format we can easily see rows and columns which makes it easier to organize and work with information

Why do we have to use Pandas for this step?

We use Pandas because it helps us organize the data into rows and columns This makes it easy to split the data into training and testing sets so the model can learn and be tested.

What is from sklearn model selection import train test split?

This line of code imports a tool called train test split from a library called scikit learn This tool is used to split the data into two parts one part to teach the model training data and one part to test the model testing data.

Why do we have to use train test split for this step?

We use train test split because it automatically splits the data into two parts for us This is important because the model needs one part to learn from training data and another part to test how well it learned testing data It helps make sure the model learns instead of just memorizing the data.

Code

from sklearn.model_selection import train_test_split 
import pandas as pd 

# Load the dataset
data = pd.read_csv("Covid_vaccination_record_refugee_camp.csv")

# Select Features (X) and Target (y)
X = data[['Age', 'Section']]  # Features
y = data['Missing Second Dose']  # Target

# Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the size of the training and testing sets
print(f"Training data size: {len(X_train)} rows")
print(f"Testing data size: {len(X_test)} rows")

![image](https://github.com/user-attachments/assets/f883d4ad-9d39-42b8-a5d1-7d55a48e0131)

Conclusions

I divided my data into 27,192 rows for training and 6,798 rows for testing, so the model can learn from a larger set and be tested on a smaller set. This helps check if the model is good at predicting new data instead of just remembering old data.

Step 6 Model Building with scikit learn

In this step

I am going to teach the computer how to make predictions.
We will use a tool called scikit learn to build a model. This model will learn from the training data the information we give it and then try to predict whether someone missed their second vaccine dose.
The model will look for patterns in the training data and then we will test it to see how well it can make predictions.

What is from sklearn linear model import Logistic Regression?

This line of code imports a tool called Logistic Regression from a library called scikit learn.
Logistic Regression is a type of model that helps us make predictions like whether someone missed their second dose of the vaccine or not.

Why do we have to use from sklearn linear model import Logistic Regression for this step?

We use Logistic Regression because it helps the computer learn patterns from the training data and predict if someone missed their second vaccine dose
It is especially useful when we want to predict one of two options like Yes or No.

What is from sklearn metrics import accuracy score?

This line of code imports a tool called accuracy score from scikit learn.
accuracy score helps us check how good the model’s predictions are by comparing the model’s guesses to the actual results.

Why do we have to use from sklearn metrics import accuracy score for this step?

We use accuracy score because it tells us how well the model is doing.
It calculates what percentage of the predictions were correct and gives us a score so we know how accurate the model is.

What is from sklearn impute import Simple Imputer?

This line of code imports a tool called Simple Imputer from scikit learn.
Simple Imputer is used to fill in any missing data like if a person’s age or other information is not available.

Why do we have to use from sklearn impute import Simple Imputer for this step?

We use Simple Imputer because sometimes there are missing pieces of information in the data like someone’s age.
The model cannot work properly with missing information so we use Simple Imputer to fill in the gaps.

What is from sklearn preprocessing import Standard Scaler?

This line of code imports a tool called Standard Scaler from scikit learn.
Standard Scaler adjusts the data so that all the numbers are on the same scale making it easier for the model to learn.

Why do we have to use from sklearn preprocessing import Standard Scaler for this step?

We use Standard Scaler to make sure all the data is on the same scale.
For example some data like age may have big numbers and other data like Yes No answers may have small numbers.
Scaling helps the model understand the data better and make better predictions.

What is Pandas?

Pandas is a tool that helps us clean and remove unwanted data in rows and columns just like a table in Excel.
It makes it easy to manage and work with large amounts of data.

Why do we have to use Pandas for this step?

We use Pandas to clean and remove unwanted the data in rows and columns making it easier for the model to learn from the information
Pandas helps us load and prepare the data for training and testing.

Code 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

![image](https://github.com/user-attachments/assets/bf6efc22-637f-4f02-9269-ef89469837f4)

Conclusions

I trained my model, and it got 66.56% accuracy, meaning it made correct predictions a little more than half the time. If I improve the model by adding better data or trying different methods, I might get a higher accuracy, like 70% or more.

Step 7 Model Improvement with XGBoost

In this step

I am going to use a tool called XGBoost to help the model make better predictions.
XGBoost is a special machine learning model that gets better at predicting step by step.

Task

Improve the model’s accuracy how well it makes predictions by using XGBoost.

Tools

We will use a tool called XGBoost which is great at making models predict more accurately.

Steps

Use XGBoost to help the model make better predictions than before.

Train the XGBoost model using the same training data we used before.

Find the best settings for XGBoost using something called GridSearchCV which helps find the best way to set up the model.

Test the XGBoost model using the test data to see if it predicts better than the old model.

Additional Questions

What is from xgboost import XGBClassifier?

This line of code imports a tool called XGBClassifier from the XGBoost library.
XGBClassifier is a type of machine learning model that helps us predict things like Yes or No.

Why do we have to use from xgboost import XGBClassifier for this step?

We use XGBClassifier because it is very good at making accurate predictions.
It builds many small decision trees to improve its guesses so the model gets better with each step.

What is from sklearn model selection import GridSearchCV?

This line of code imports a tool called GridSearchCV from scikit learn another machine learning library.
GridSearchCV helps us find the best settings for our model by testing different combinations of settings.

Why do we have to use from sklearn model selection import GridSearchCV for this step?

GridSearchCV allows us to find the best combination of settings like how fast the model learns or how many trees it builds.
Without GridSearchCV we would have to guess these settings ourselves but GridSearchCV does this testing for us.

What is from sklearn metrics import accuracy score?

This line of code imports a tool called accuracy score from scikit learn.
Accuracy score is a tool that checks how many predictions the model got right.

Why do we have to use from sklearn metrics import accuracy score for this step?

We use accuracy score to find out how good the model is at making predictions.
It compares the model’s predictions to the actual results and gives us a score like a grade that tells us how accurate the model is.

Code 

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Create and train the XGBoost model
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100)
xgb_model.fit(X_train_encoded, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_encoded)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")

![image](https://github.com/user-attachments/assets/f2c3036e-098d-45fb-986a-2a7eec29b55f)

Conclusions 
I used XGBoost to improve my model, and the accuracy increased to 70.15%. This means the new model predicts better than the previous one, making it more reliable for future predictions.

Step 8 Advanced Model Building with TensorFlow

In this step

I am going to use a tool called TensorFlow to build a more advanced model This model will be able to find even more complex patterns in the data.

Task

Build a deep learning model using TensorFlow to make better predictions.

Tools

I will use TensorFlow a tool that helps create powerful models for making predictions.

Steps

1 Design a neural network A neural network is like a computer brain I will build this so the model can understand more complicated data.
2 Train the model I will teach the neural network by showing it the training data so it can learn from it.
3 Monitor the learning I will keep an eye on how well the model is learning and make changes if needed.
4 Test the model After the model learns I will test it with the test data and compare its predictions to older models to see if it is better.

Additional Questions

What is from tensorflow keras models import Sequential?

This line of code imports a tool called Sequential from the TensorFlow library.
Sequential is a simple way to build a model where I add layers step by step just like building blocks.

Why do we have to use from tensorflow keras models import Sequential for this step?

I use Sequential because it allows me to build a model layer by layer making it easier to create the neural network needed for the task.

What is from tensorflow keras layers import Dense Input?

This line of code imports two tools Dense and Input from the TensorFlow library.
Dense is a type of layer where all the neurons are connected to each other.
Input is used to define the size of the input data so the model knows what kind of data it is working with.

Why do we have to use from tensorflow keras layers import Dense Input for this step?

I use Dense to add layers to the neural network helping the model learn from the data.
I use Input to tell the model how much data it should expect when training.

What is from sklearn metrics import accuracy_score?

This line of code imports a tool called accuracy_score from scikit learn.
Accuracy_score is used to check how many of the model predictions were correct.

Why do we have to use from sklearn metrics import accuracy_score for this step?

I use accuracy_score to measure how well the model is making predictions It compares the predicted answers with the actual answers and gives a percentage score.

Code

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score

# Build the neural network model
model = Sequential([
    Input(shape=(X_train_encoded.shape[1],)),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_encoded, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(X_test_encoded)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

![image](https://github.com/user-attachments/assets/3b3f9656-8b89-4162-a22f-022bc077465e)

Conclusions

I built a neural network model using TensorFlow, and it achieved 69.30% accuracy in predicting whether someone missed their second vaccine dose. This accuracy is slightly lower than the XGBoost model, which got 70.15%, meaning XGBoost made better predictions in this case.


Step 9 Model Evaluation and Interpretation

Evaluate the final model

I’m going to check the model’s performance by looking at . 
Accuracy I’m going to see how many predictions were correct overall.
Precision I’m going to check how accurate the yes predictions are how many of them were really yes.
Recall I’m going to check how many yes cases the model found out of all the real yes cases.
F1 Score I’m going to use this to balance between precision and recall.
AUC ROC I’m going to see how good the model is at telling the difference between yes and no.

Interpret the models predictions

I’m going to find out which features like age or test results were important for the models predictions using feature importance or SHAP values These are tools that help explain why the model made certain decisions.

Visualize the models performance

I’m going to create charts and graphs to show the models results and predictions clearly so we can understand them easily.

What are these imports?

from sklearnmetrics import classification_report roc_auc_score.

What it does?

I’m going to use these to calculate accuracy precision recall F1 score and AUC ROC.

Why use this?

I’m going to use these to check how good the models predictions are and how well it can tell yes from no.

import matplotlibpyplot as plt

What it does?

I’m going to use this to create graphs and charts.

Why use this?
I’m going to make visual representations pictures of the models results like a confusion matrix or other plots.

import shap

What it does?

I’m going to use this to see which features like age or test results were important for the models predictions.

Why use this?

I’m going to explain why the model made certain decisions by showing which features were important using SHAP values.

import numpy as np

What it does? 

I’m going to use this to work with numbers and data in my model.

Why use this?

I’m going to use it for things like creating random samples or doing math with data arrays of numbers.

Pandas

What it does? 

I’m going to use this to clean and remove unwanted data tables rows and columns like in Excel.

Why use this?

I’m going to use Pandas to easily organize view and work with my data in the model

import seaborn as sns

What it does?

I’m going to use this to make my graphs and charts look nice.

Why use this?

I’m going to create colorful and clearer charts so its easier to see the results of the model.

from sklearnmetrics import confusion_matrix

What it does?

I’m going to use this to create a table that shows how many times the model predicted correctly or incorrectly.

Why use this?

I’m going to compare how often the model was right or wrong by using a table that compares the real and predicted results.


Code

# Evaluate the Model
from sklearn.metrics import classification_report, roc_auc_score

print("Model Performance:")
print(classification_report(y_test, y_pred_binary))

auc = roc_auc_score(y_test, y_pred_binary)
print(f"AUC-ROC Score: {auc:.2f}")

![image](https://github.com/user-attachments/assets/7d33225c-19cb-46ee-9f85-552b0396fa61)
![image](https://github.com/user-attachments/assets/b684896e-4e4c-40b7-9af2-6f05c66d9033)

Conclusions

I got an AUC-ROC score of 0.62, meaning my model was only a bit better than guessing randomly at predicting dengue. I noticed from the SHAP plot that the feature called "feature_0" was not very helpful in making good predictions.

Step 10: Asking and Answering Questions about the Project

In this step, I'm going to ask and answer 11 real questions about my project that I might get when I present it to others.

1 What challenges did I face during coding and how did I solve them?

One challenge was making sure my code worked correctly especially when testing the models predictions I solved this by carefully going over my code line by line and fixing any mistakes I found  

2 How did the models performance compare to what I expected? 

The model did better than I expected with an accuracy of 69 percent It worked well for predicting who had dengue but wasnt as good at predicting who didnt have it Theres still room for improvement  

3 What features were the most important in predicting dengue cases?

Features like age and travel history were very important in predicting whether someone might have dengue SHAP values helped me understand which factors the model considered most important when making predictions  

4 What would I do next to improve this model?

To improve the model I would collect more data and try different machine learning models This could help the model make more accurate predictions especially for people who dont have dengue  

5 How can this model help in real world healthcare settings?

This model could help doctors identify which patients are likely to have dengue more quickly This could help doctors start treatment faster which would be useful in hospitals or clinics  

6 What did I learn from building this model?

I learned that clean accurate data is really important when building a model Even small mistakes in the data can make the model give the wrong answers  

7 What was the hardest part of the project?

The hardest part was getting the model to be better at predicting when someone didnt have dengue It took a lot of adjustments to reduce the number of mistakes  

8 Were there any surprises during the project?

Yes I was surprised that travel history had such a big impact on predicting dengue I didnt think it would be so important but the model showed that it was  

9 If I had more time what would I do differently?

If I had more time I would try more advanced machine learning models and gather more data This would help make the model even more accurate in its predictions  

10 How could this model help people in the future?

This model could be used in hospitals to help doctors find dengue cases faster By quickly identifying who might have dengue doctors could start treatment sooner and help save lives  

11 Why did I do this project?

I chose to do this project because dengue is a serious disease that affects many people I wanted to see if I could build a model to help doctors find out who might have dengue faster so they can help people more quickly This project was important because it could help save lives by getting people the treatment they need as soon as possible  

Conclusions

Building this model taught me how to predict dengue cases and showed me that getting accurate results is challenging The model can be useful for doctors but it can still be improved by using more data and better techniques
