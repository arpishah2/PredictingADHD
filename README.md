# PredictingADHD

#Goal
Detect ADHD by using machine learning binary classification technique

#Dataset information
Dataset consists of attributes out of which attributes having primary importance are VAERS_ID and symptoms. The VAERS identification number is a sequentially assigned number used for identification purposes. MedDRA Terms are derived from this symptom text and placed in the VAERS Symptoms.
Example symptom: 12 hrs p/vax pt was having severe pain & swelling of left arm; inflammation inc in size from elbow to back of scapula, (red, & hot to touch); family MD treated w/DPh & PCN & APAP & ice packs;

#Models Used:
Naive Bayes and Logistic Regression

#Conclusion:
Naive Bayes and Logistic Regression models have been built in Weka and Apache Spark. Model based on logistic regression performs better than model built using Naive Bayes classifier. Hence, if the model is used in real world scenario then it can prove to solve some diagnostic problems of ADHD.


