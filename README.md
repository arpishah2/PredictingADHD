# PredictingADHD

#Goal
Detect ADHD by using machine learning binary classification technique

#Dataset information
Dataset consists of attributes out of which attributes having primary importance are VAERS_ID and symptoms. The VAERS identification number is a sequentially assigned number used for identification purposes. MedDRA Terms are derived from this symptom text and placed in the VAERS Symptoms.
Example symptom: 12 hrs p/vax pt was having severe pain & swelling of left arm; inflammation inc in size from elbow to back of scapula, (red, & hot to touch); family MD treated w/DPh & PCN & APAP & ice packs;

#Models Used:
Naive Bayes and Logistic Regression

#Note
Please view "Predicting Attention Deficit Hyperactive Disorder using large s.pdf" for detailed information

#Conclusion:
Naive Bayes and Logistic Regression models have been built in Weka and Apache Spark. Model based on logistic regression performs better than model built using Naive Bayes classifier. Hence, if the model is used in real world scenario then it can prove to solve some diagnostic problems of ADHD.

Error
remote: warning: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: warning: See http://git.io/iEPt8g for more information.
remote: warning: File SparkData/negative/newfile.csv is 66.97 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB

