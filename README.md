# naive-bayes-classification-implementation
Implementation of Naive Bayes Multi-Class Classification Algorithm with attaining accuracy 95.38 %

The given classification problem is implemented on a given dataset which is described as "The dataset for this problem
is a subset of the Reuters-21578 dataset and has been obtained from this website (look at the R8 dataset).
Read the website for more details about the dataset. You have been provided with separate training and
test files containing 5485 and 2189 articles (examples), respectively. Each article comes from one of the
eight categories (class label). Each row in the file contains the information about the class of the article
followed by the list of words appearing in the article".

I have commented in a very better manner that even a novice will be able to understand the code and used the variable name accordingly so that there is no problem "What the heck is this variable (so many variables one can miss)".

Running it 

First run the file train.py which will generate some files which will be useful to the test.py file and further test files will generate some files which will be used by accuracy.py to check the accuracy.
Generation of the files has been done because reading and extracting data again in the test file will again take time.
I have counted the number of occurances a word is having in a line while classification.

I have also run the ML Package 'skicit-learn' and i achieved the same accuracy on this dataset to the sixth decimal places.
