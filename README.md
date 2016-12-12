# CSCI567_mini_project
#Dependencies
Project depends on the following libraries:

1. pandas: http://pandas.pydata.org/pandas-docs/version/0.17.1/install.html

2. Keras with Theano: https://keras.io/#installation

3. Scikit-Learn: http://scikit-learn.org/stable/install.html

4. XGBoost: http://xgboost.readthedocs.io/en/latest/build.html

5. Gensim: https://radimrehurek.com/gensim/install.html

# Repository Overview:

* Data folder contains all the dataset files.

* The purpose of ndcg.py is to calculate the NDCG score.

* Utils.py contains the implementation of Deepnet interface. It was provided to us as part of the class homework#4

* xgb_local_validation.py implements the XGB classifier. It splits the training set into 80:20 subparts and uses it as a validation set to predict the NDCG score. Henceforth we will call this local validation. It uses the F score as a feature selection method.

* xgb_gini_local_validation.py is same as xgb_local_validation.py except it uses Gini index for feature selection.

* xgb_bytecup_validation.py implements the XGB classifier. It uses the entire training dataset and generates the temp.csv having all the probabilities. This CSV file can be submited to Bytecup 2016 competition and get the NDCG score. Henceforth we will refer this as bytecup validation as opposed to local one described above.

* deepnet_bytecup_validation.py implements the Deep Neural Net using Keras Library with Theano backend. It generates the output file for bytecup validation.

* voting_classifier.py implements the Voting Classifier ensemble technique and generates the output CSV. This can be submitted to Bytecup.

