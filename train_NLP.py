import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
import joblib
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class SentimentAnalysisModel:
    def __init__(self, train_directory, model_file_path, vectorizer_file_path):
        # Initialize class variables
        self.train_directory = train_directory
        self.reviews_train = []
        self.labels_train = []
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.max_features_selected = 0
        self.best_model_max_features_selected = 0
        self.vocab_size = 0
        self.best_model_vocab_size = 0
        self.vectorizer = None
        self.best_model_vectorizer = None
        self.model = None
        self.best_model = None
        self.model_history = None
        self.best_model_history = None
        self.model_file_path = model_file_path
        self.vectorizer_file_path = vectorizer_file_path

    def load_dataset(self, directory):
        # Load the dataset and labels from the specified directory
        reviews = []
        labels = []
        for sentiment in ['pos', 'neg']:
            sentiment_directory = directory + sentiment + "/"
            for file_name in os.listdir(sentiment_directory):
                with open(sentiment_directory + file_name, 'r') as file:
                    review = file.read()
                    reviews.append(review)
                    labels.append(1 if sentiment == 'pos' else 0)
        return reviews, labels

    def preprocess_reviews(self, reviews):
        # Preprocess the reviews using tokenization, lemmatization, and stop-word removal
        processed_reviews = []
        for review in reviews:
            tokens = word_tokenize(review)
            filtered_tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words]
            processed_review = ' '.join(filtered_tokens)
            processed_reviews.append(processed_review)
        print("Sample Review before processing")
        print(reviews[0])
        print("Sample Review before processing")
        print(processed_reviews[0])
        return processed_reviews

    def set_vocab_size(self, reviews):
        # Calculate the vocabulary size from the given reviews
        unique_words = set()
        for review in reviews:
            tokens = word_tokenize(review)
            unique_words.update(tokens)
        self.vocab_size = len(unique_words)

    def fit_vectorizer(self, X_train):
        # Fit the TfidfVectorizer on the training data
        self.set_vocab_size(X_train)
        self.max_features_selected = (20 * self.vocab_size) // 100 # Pareto principle
        self.vectorizer = TfidfVectorizer(max_features=self.max_features_selected)
        self.vectorizer.fit(X_train)
        print("Vocabulary Size of processed data: ", self.vocab_size)
        print("Maximum Features Selected for TF-IDF Vectorizer: ", self.max_features_selected)

    def train_classifier(self, X_train_tfidf, y_train, X_val_tfidf, y_val):
        # Train the sentiment analysis model using the specified architecture
        self.model = Sequential()
        l2_penalty = 0.01

        self.model.add(Dense(512, input_dim=X_train_tfidf.shape[1], activation='relu', kernel_regularizer=l2(l2_penalty)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu', kernel_regularizer=l2(l2_penalty)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_penalty)))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.summary()
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        self.model_history = self.model.fit(X_train_tfidf.toarray(), np.array(y_train), epochs=20, batch_size=2048, validation_data=(X_val_tfidf.toarray(), np.array(y_val)))

    def train_and_save_model(self, num_folds=5):
        # Load the dataset, preprocess it, and train the model using K-fold cross-validation
        self.reviews_train, self.labels_train = self.load_dataset(self.train_directory)
        self.reviews_train = self.preprocess_reviews(self.reviews_train)

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        val_accuracies = []
        best_val_accuracy = 0
        for (train_idx, val_idx) in skf.split(self.reviews_train, self.labels_train):
            X_train_fold, X_val_fold = np.array(self.reviews_train)[train_idx], np.array(self.reviews_train)[val_idx]
            y_train_fold, y_val_fold = np.array(self.labels_train)[train_idx], np.array(self.labels_train)[val_idx]

            self.fit_vectorizer(X_train_fold)

            X_train_tfidf = self.vectorizer.transform(X_train_fold)
            X_val_tfidf = self.vectorizer.transform(X_val_fold)

            self.train_classifier(X_train_tfidf, y_train_fold, X_val_tfidf, y_val_fold)

            val_accuracy = round(self.model_history.history['val_accuracy'][-1]*100,2)
            val_accuracies.append(val_accuracy)

            # Save the best accuracy model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.best_model = self.model
                self.best_model_vectorizer = self.vectorizer
                self.best_model_history = self.model_history.history
                self.best_model_vocab_size = self.vocab_size
                self.best_model_max_features_selected = self.max_features_selected

        print("Vocabulary Size of Best model: ", self.best_model_vocab_size)
        print("Maximum Features Selected for TF-IDF Vectorizer in best model: ", self.best_model_max_features_selected)
        
        for index, val_accuracy in enumerate(val_accuracies):
            print(index+1, " Fold Accuracy = ", val_accuracy)
        
        print("Best Validation accuracy: ", best_val_accuracy)
        print("Average Validation accuracy: ", round(sum(val_accuracies)/len(val_accuracies),2))
        print("Saving the best model and vectorizer")
        self.save_model_and_vectorizer()

    def save_model_and_vectorizer(self):
        # Save the best model and vectorizer to disk
        self.best_model.save(self.model_file_path)
        joblib.dump(self.best_model_vectorizer, self.vectorizer_file_path)

    def plot_training_and_validation_metrics(self):
        # Plot the training and validation metrics (accuracy and loss)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.best_model_history['accuracy']) + 1), self.best_model_history['accuracy'], label='Training Accuracy')
        plt.plot(range(1, len(self.best_model_history['val_accuracy']) + 1), self.best_model_history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Average Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.best_model_history['loss']) + 1), self.best_model_history['loss'], label='Training Loss')
        plt.plot(range(1, len(self.best_model_history['val_loss']) + 1), self.best_model_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Average Training and Validation Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    # Main function to train and evaluate the model
    train_directory = "./data/aclImdb/train/" # train directory path
    model_file_path = "./models/NLP_model.h5" # model file path to save
    vectorizer_file_path = './models/vectorizer.joblib' # vectorizer file path to save

    model = SentimentAnalysisModel(train_directory, model_file_path, vectorizer_file_path)
    model.train_and_save_model(num_folds=5)
    model.plot_training_and_validation_metrics()

if __name__ == '__main__':
    main()