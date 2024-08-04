import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix , recall_score, f1_score, classification_report
from matplotlib import pyplot as plt
import seaborn as sb

#   Data
data = pd.read_csv('emails.csv')
print(data.head())
data.info()

print(data.isnull().sum())
print(data.shape)

new_df=data.dropna()
print (new_df)

#   Model Training
data_drop=data.drop('Email No.', axis = 1)
X= data_drop.drop('Prediction', axis=1)
Y=data_drop['Prediction']
print('Shape of X', X.shape)
print('Shape of Y', Y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=44)
classifier =  MultinomialNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#   Model Evaluation
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
