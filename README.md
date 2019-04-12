# NLP_Movie_Reviews
In this report we are going to predict the sentiments of the movie reviews with the help of NLP tools and XGboost classifier.

# Tools/Library
Tools Used: Spyder, Notepad++, Ms Excel
Libraries Used: pandas, regex, nltk: (porterstemmer, stopwords), scikit learn: (cross_val_score, train_test_split, confusion_matrix, TfidfVectorizer), XGBoost: XGBClassifier

# Cleaning & Pre-processing 
The raw text is pretty messy for these reviews so before we can do any analytics we need to clean things up: duplication removal etc.

# Vectorization
In order for this data to make sense to our machine learning algorithm we’ll need to convert each review to a numeric representation, which we call vectorization.

# Build Classifier 
In our case XGBoost was used. 

# Apart from this we have used other sophisticated methods : 
1.Text Processing: Stemming

2.n-grams: Instead of just single-word tokens (1-gram/unigram) we can also include word pairs(bi-grams).

3.Representations: Instead of simple, binary vectors we can use TF-IDF to transform those counts.
