#importing required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report


#after importing the libraries we will make a small datasets of emails
emails = [
    "Get rich quick ! Click here to win a million dollars",
    "hello coud you review this document for me ?",
    "DISCOUNT SALE ! Get 50  percent off on all products from our side",
    "GET your money double in 24 hours !",
    "Meeting appointment at 3 pm tomorrow",
    "Congratulations ! you won a free gift"
]
labels=[1,0,1,0,1,1]
#convert text data into numerical features using count vectorization 
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

#split data into training ans testing sets 
X_train , X_test ,y_train , y_test = train_test_split(x,labels,test_size=0.2)

#create a multinomial Naive Bayes classifier
model =MultinomialNB()


#train the model on the training data
model.fit(X_train,y_train)

#make predicition on test data
y_pred = model.predict(X_test)


#Evaluate the model 
accuracy=accuracy_score(y_test,y_pred)
report= classification_report(y_test,y_pred)

print("Accuracy:",accuracy)
print("Classification Report:")
print(report)


#predict whether a new email is spam or not
new_email = "Congratulations ! you won a free gift"
new_email_vectorized = vectorizer.transform([new_email])
prediction_label = model.predict(new_email_vectorized)

if prediction_label[0] == 0:
    print("The email is not spam")
else:
    print("The email is spam.")
