import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """
    A class to perform feature extraction on preprocessed text data using
    GloVe embeddings and TF-IDF weighting.

    This class encapsulates the pipeline from the feature_extraction notebook.
    It loads a pre-trained GloVe model and the preprocessed train/test dataframes,
    calculates TF-IDF weights from the training set, creates a weighted average
    GloVe vector for each user, and saves the resulting dataframes.
    """

    def __init__(self, glove_path, train_data_path, test_data_path, output_dir):
        """
        Initializes the FeatureExtractor with necessary file paths.

        Args:
            glove_path (str): Path to the GloVe word embeddings file.
            train_data_path (str): Path to the preprocessed 'train.pkl' file.
            test_data_path (str): Path to the preprocessed 'test.pkl' file.
            output_dir (str): Directory to save the final feature-extracted .pkl files.
        """
        self.glove_path = glove_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.output_dir = output_dir

        self.glove_model = None
        self.embedding_dim = None
        self.word_idf_weights = None
        self.train_df = None
        self.test_df = None

        # Initialize tqdm for pandas operations
        tqdm.pandas()

    def _load_glove_model(self):
        """
        Loads the GloVe word vectors from the specified file path into a dictionary.
        Also determines and sets the embedding dimension.
        """
        print(f"1. Loading GloVe model from {self.glove_path}...")
        self.glove_model = {}

        file_size = os.path.getsize(self.glove_path)

        with open(self.glove_path, 'r', encoding="utf-8") as f:
            with tqdm(total=file_size, unit='B', unit_scale=True,
                      desc=f"Loading {os.path.basename(self.glove_path)}") as pbar:
                for line in f:
                    split_line = line.split()
                    word = split_line[0]
                    embedding = np.array([float(val) for val in split_line[1:]])
                    self.glove_model[word] = embedding

                    if self.embedding_dim is None:
                        self.embedding_dim = len(embedding)

                    pbar.update(len(line.encode('utf-8')))

        print(f"Loading complete. {len(self.glove_model)} words loaded with embedding dimension {self.embedding_dim}.")
        print("-" * 30)

    def _load_preprocessed_data(self):
        """
        Loads the preprocessed training and test data from .pkl files.
        """
        print("2. Loading preprocessed data...")
        self.train_df = pd.read_pickle(self.train_data_path)
        print(f"Loaded 'train.pkl' with {len(self.train_df)} rows.")
        self.test_df = pd.read_pickle(self.test_data_path)
        print(f"Loaded 'test.pkl' with {len(self.test_df)} rows.")
        print("-" * 30)

    def _calculate_tfidf_weights(self):
        """
        Calculates TF-IDF (specifically IDF) weights.

        IMPORTANT: The TfidfVectorizer is fitted ONLY on the training data to prevent
        data leakage from the test set. The learned IDF weights are then used for
        both training and test feature extraction.
        """
        print("3. Calculating TF-IDF weights from training data...")

        # Join the list of words into a single string for the vectorizer
        train_text = self.train_df['aggregated_words'].str.join(' ')

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(train_text)

        self.word_idf_weights = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
        print("TF-IDF weights calculated and stored.")
        print("-" * 30)

    def _create_tfidf_weighted_vector(self, word_list):
        """
        Calculates the TF-IDF weighted mean GloVe vector for a list of words.
        """
        vectors = []
        weights = []

        for word in word_list:
            if word in self.glove_model and word in self.word_idf_weights:
                vectors.append(self.glove_model[word])
                weights.append(self.word_idf_weights[word])

        if not vectors:
            return np.zeros(self.embedding_dim)

        return np.average(vectors, axis=0, weights=weights)

    def _apply_feature_creation_and_save(self):
        """
        Applies the feature creation process to both training and test sets
        and saves the results.
        """
        print("4. Creating TF-IDF weighted feature vectors and saving datasets...")

        # --- Process Training Data ---
        print("Processing training set...")
        self.train_df['fv'] = self.train_df['aggregated_words'].progress_apply(self._create_tfidf_weighted_vector)

        # --- Process Test Data ---
        print("\nProcessing test set...")
        self.test_df['fv'] = self.test_df['aggregated_words'].progress_apply(self._create_tfidf_weighted_vector)

        # --- Save Data ---
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"\nCreated output directory: {self.output_dir}")

        train_save_path = os.path.join(self.output_dir, 'train.pkl')
        test_save_path = os.path.join(self.output_dir, 'test.pkl')

        self.train_df.to_pickle(train_save_path)
        print(f"Training set with feature vectors saved to {train_save_path}")

        self.test_df.to_pickle(test_save_path)
        print(f"Test set with feature vectors saved to {test_save_path}")
        print("-" * 30)

    def extract_features(self):
        """
        Executes the entire feature extraction pipeline in order.
        """
        self._load_glove_model()
        self._load_preprocessed_data()
        self._calculate_tfidf_weights()
        self._apply_feature_creation_and_save()
        print("\nFeature extraction pipeline finished successfully.")


if __name__ == '__main__':
    # --- Configuration ---
    BASE_DATA_DIR = '../data'
    PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'processed')
    EXTRACTED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'extracted')
    GLOVE_DIR = os.path.join(BASE_DATA_DIR, 'GloVe')

    # Define paths to the required files
    glove_file_path = os.path.join(GLOVE_DIR, 'glove.twitter.27B.50d.txt')
    train_pkl_path = os.path.join(PROCESSED_DATA_DIR, 'train.pkl')
    test_pkl_path = os.path.join(PROCESSED_DATA_DIR, 'test.pkl')

    # --- Execution ---
    # Check if required files exist before running
    required_files = [glove_file_path, train_pkl_path, test_pkl_path]
    if not all(os.path.exists(f) for f in required_files):
        print(
            "Error: One or more required data files not found. Please check the paths in the `if __name__ == '__main__':` block.")
    else:
        # Instantiate the extractor
        extractor = FeatureExtractor(
            glove_path=glove_file_path,
            train_data_path=train_pkl_path,
            test_data_path=test_pkl_path,
            output_dir=EXTRACTED_DATA_DIR
        )

        # Run the entire feature extraction pipeline
        extractor.extract_features()