import os
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class SentimentAnalysisModel:
    def __init__(self, model_file, vectorizer_file):
        # Initialize the SentimentAnalysisModel with the trained model and vectorizer
        self.model = load_model(model_file)
        self.vectorizer = joblib.load(vectorizer_file)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_reviews(self, reviews):
        # Preprocess the reviews using tokenization, lemmatization, and stop-word removal
        processed_reviews = []
        for review in reviews:
            tokens = word_tokenize(review)
            filtered_tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words]
            processed_review = ' '.join(filtered_tokens)
            processed_reviews.append(processed_review)
        return processed_reviews

    def load_test_data(self, test_directory):
        # Load the test data and labels from the specified test directory
        reviews = []
        labels = []
        for sentiment in ['pos', 'neg']:
            sentiment_directory = test_directory + sentiment + "/"
            for file_name in os.listdir(sentiment_directory):
                with open(sentiment_directory + file_name, 'r') as file:
                    review = file.read()
                    reviews.append(review)
                    labels.append(1 if sentiment == 'pos' else 0)
        return reviews, labels

    def load_and_preprocess_data(self, test_directory):
        # Load and preprocess the test data
        self.reviews_test, self.labels_test = self.load_test_data(test_directory)
        self.reviews_test = self.preprocess_reviews(self.reviews_test)

    def predict_output(self, X_input):
        # Predict the sentiment labels for the given input using the trained model
        X_input_tfidf = self.vectorizer.transform(X_input)
        y_pred_probs = self.model.predict(X_input_tfidf.toarray())
        y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_probs]
        return y_pred

    def evaluate(self, y_true, y_pred):
        # Evaluate the accuracy of the model's predictions
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def run_test_data(self, test_directory):
        # Load and preprocess the test data, predict the labels, and evaluate the model's performance
        self.load_and_preprocess_data(test_directory)
        y_test_pred = self.predict_output(self.reviews_test)
        accuracy_test = self.evaluate(self.labels_test, y_test_pred)
        print(f"Test Accuracy: {accuracy_test*100:.2f}%")

def main():
    # Main function to load the trained model and vectorizer, and run test data
    model_file = './models/NLP_model.h5' # model file path
    vectorizer_file = './models/vectorizer.joblib' # vectorizer file path
    test_directory = './data/aclImdb/test/' # test directory path

    model = SentimentAnalysisModel(model_file, vectorizer_file)
    model.run_test_data(test_directory)

if __name__ == '__main__':
    main()