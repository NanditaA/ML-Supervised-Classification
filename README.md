# ML-Supervised-Classification
Using Machine Learning Techniques to predict the Recurrence of Breast Cancer
reast Cancer affects close to 2 million women globally every year and is a leading cause of death. Fortunately with advancements in treatments and diagnostic ability, early detection of Breast Cancer allows physicians to treat the disease aggressively giving women a better chance of survival.

Often following the diagnosis and treatment of the primary incidence of breast cancer, recurrence events are seen. A recurrence event is characterised by the cancer “coming back” after at least a year of remission either locally (in the same organ), regionally(in a close-by organ) or distant (in another part of the body). Early detection of recurrence events (i.e. while still asymptomatic) is more likely to be curable rather than when a women seeks treatments after the cancer symptoms are seen again.

What makes Machine Learning so Critical in Breast Cancer Treatment
The current model of breast cancer follow-up care is retrospective, time consuming and expensive, as it involves a multi-disciplinary approach requiring clinicians, radiographers, surgeons. Patients regularly meet with an oncologist to identify new symptoms or changes on a physical exam. This ‘traditional’ method of diagnosis can often be retrospective as recurrence events are picked up after symptoms reoccur leading to poorer outcomes.Surveillance methods like Mammography, PET, Breast MRI, Ultrasound are expensive and their efficacy is as yet clinically unproven via randomised controlled trials. Hence their use is reserved for symptomatic patients with highest risk of recurrence.

Advances in the field of Machine Learning have enabled us to apply it to address the above mentioned gaps in the existing model of care. Machine Learning techniques can be used to identify women at risk of a recurrence using known gynaecological metrics and other cancer features. This method is non-invasive, cost effective, can be used to proactively identify asymptomatic women at risk of cancer recurrence. Machine Learning based surveillance programs can be used to supplement the existing model of care for adjuvent Breast Cancer treatment with an aim to improve survival rates by optimising treatments.

The Breast Cancer Dataset:
This article evaluates several Machine Learning Classification Algorithms to predict the recurrence of Breast Cancer.

We used the Breast Cancer Data Set with 286 instances of real patient data obtained from the Institute of Oncology, Ljubljana. The data set is publicly available from UCI Machine Learning repository and has the following variables:

Age: age of the patient at the time of diagnosis;
Menopause: whether the patient is pre- or postmenopausal at time of diagnosis;
Tumor size: the greatest diameter (in mm) of the excised tumor;
Inv-nodes: the number (range 0 - 39) of axillary lymph nodes that contain metastatic breast cancer visible on histological examination;
Node caps: if the cancer does metastasise to a lymph node, although outside the original site of the tumor it may remain “contained” by the capsule of the lymph node. However, over time, and with more aggressive disease, the tumor may replace the lymph node and then penetrate the capsule, allowing it to invade the surrounding tissues;
Degree of malignancy: the histological grade (range 1-3) of the tumor. Tumors that are grade 1 predominantly consist of cells that, while neoplastic, retain many of their usual characteristics. Grade 3 tumors predominately consist of cells that are highly abnormal;
Breast: breast cancer may obviously occur in either breast;
Breast quadrant: the breast may be divided into four quadrants, using the nipple as a central point;
Irradiation: radiation therapy is a treatment that uses high-energy x-rays to destroy cancer cells. 
Data Cleaning & Preparation:
Converting Strings to Numerical Values:
Most variables in the data set were provided as strings and had to be converted to a numeric representation for analysis.

Age, Tumor Nodes & Tumor Size were presented in ranges eg. (24-29) etc. For the purpose of modelling, the numerical average of these ranges were computed and used in every case. While this simplified the analysis, it under-represented higher values in the range. This phenomenon is expected to decrease the quality of the model as we expect the size and presence of tumours and higher ages to be strong risk factors to the recurrence of breast cancer.
Other variables e.g. Irradiation, Breast, Breast-quadrant, menopause, degree of malignancy, presence of node caps (Y/N) were either binarized (in case of 2 variables) or were provided distinct numerical variables to categorise them numerically. 
Deleting Erroneous Data:
There was a disproportionate number of cases showing right-breast quadrants affected by cancer (expecting roughly 50:50 split). On investigation, we found that breast-quadrants were incorrectly assigned to “right” even though the left breast was impacted. We deleted the breast quadrant input field.
Oversampling to address Dataset Imbalance: SMOTE (Synthetic Minority Over-sampling Technique).
The Breast Cancer dataset was imbalanced with significantly more no-recurrence cases vs recurrence cases. We applied the SMOTE technique, which involved synthetically oversampling the minority class. Python has an inbuilt SMOTE function:

from imblearn.over_sampling import SMOTE
Data Visualisation:

Our Objective:
Based on the input variables listed above, we aim to build a classification model to classify patient data into 2 classes:

●Recurrence Events

●Non-Recurrence Events

A recurrence event is characterised by the cancer “coming back” after at least a year of remission. This reappearance can be either locally (in the same organ), regionally(in a close-by organ) or distant (in another part of the body).

Early detection of recurrence events (i.e. while still asymptomatic) is more likely to be curable rather than when a women seeks treatments after the cancer symptoms are seen again. Thus this classification task has high clinical significance and can be used to complement the traditional model of care.

Method:

Building the Model: Fitting on Training Data Set
#Trying Different Types of Classifiers
#1.Decision Tree Classifier
#recurrence_classifier = DecisionTreeClassifier(max_leaf_nodes=19, random_state=0)

#2. Logistic Regression
#recurrence_classifier = LogisticRegression(random_state = 0)

#3. K-Nearest Neighbours
#recurrence_classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)

#4.Support Vector Classification
#recurrence_classifier = SVC(kernel = 'linear', random_state = 0)

#4.Gaussian Naïve Bayes Algorithm
#recurrence_classifier = GaussianNB()

#5.Random Forest Algorithm
#recurrence_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

#*************FITTING CLASSIFIER ON TRAINING SET*************
recurrence_classifier.fit(features_train_res, Output_train_res)
Five (supervised) machine learning classification algorithms were trained using all inputs above to classify patient cases into 2 distinct classes, ie. Cancer Recurrence Events vs Non Recurrence Events.

Using the Classification Model to Predict Breast Cancer Recurrence:
The classifier object was then used to predict the class on the test slice of the dataset using recurrence_classifier.predict(features_test)

On testing the model accuracy of each algorithm, we found the Regression Classifier and the Gaussian Naïve Bayes Classifiers had the highest accuracy at 73%.

●#1.Decscion Tree Classifier: 64.27%
●#2.Logistic Regression: 73.17%
●#3.KNearest Neighbour: 67.54%%
●#4.Support Vector Machine Model (SVC): 70%
●#5.Gaussian Naive Bayes Algorithm: 73.17%
●#6.Random Forest Classifier: 70.21%
We recognise this accuracy level is fairly low for the model to be used reliably in a clinical setting and have identified several limitations within the dataset, which if addressed would improve the model accuracy.  

Model Limitations:
Womens age & the Number & Size of the tumor need to be provided in distinct numeric values instead of as a range as the model looses valuable granularity of these input variables. Additionally, the dataset doesn’t include all risk factors of cancer recurrence ie. Family history or breast cancer and Genetic Mutations (BRCA1 & BRCA2). Availability of these metrics would improve the prediction accuracy score.

The dataset doesn’t contain details about the cancer cell nuclei: eg. (radius, perimeter, area, texture, smoothness, compactness, concavity, symmetry etc). Subsequent data sets made available by UCI machine learning repository have this data. Utilising this information should improve model classification accuracy & precision scores.

Conclusions:
●Machine Learning algorithms can be used to supplement the existing model of care for the early detection of Breast Cancer Recurrence Events. Given these algorithms require analytical input variables & don’t rely on actual cancer symptoms, they can be used to improve patient survival outcomes at the population level in a more cost-effective manner.

●Of the various Machine Learning Algorithms investigated, the Regression Classifier and the Gaussian Naïve Bayes models had the best accuracy in predicting Recurrence Events (Class 1) vs No-Recurrence Events (class 2).

●Unfortunately due to the limitations identified in the dataset the maximum model prediction accuracy of 73% necessitates m5.ore work to be done before this model can be confidently used in a clinical setting. The risk with such a low accuracy score is the number of false positives (women with potential recurrence events that test negative) that may go undetected. 

References & Acknowledgements:
1.Current Approaches and Challenges in the Early Detection of Breast Cancer Recurrence. Journal of Cancer. March 2016.

2.American Cancer Society.

3.Breast cancer diagnostic typologies by grade-of-membership fuzzy modelling. Proceedings of the 2nd WSEAS International Conference on Multivariate Analysis and its Application in Science and Engineering .University of Lisbon,2009.

4. UCI Centre for Machine Learning and Intelligence Systems. Breast Cancer Data Set.

5. This breast cancer domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. Thanks go to M. Zwitter and M. Soklic for providing the data. It is available online here.

6. I would like to acknowledge work done by Vishabh Goel in building a simple machine learning model to predict breast cancer on a different dataset. His work provided inspiration for my project.

