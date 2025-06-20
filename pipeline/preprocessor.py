import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os


class Processor:
    """
    A class to preprocess user job data, including merging user information,
    processing text unigrams, and splitting the data into training and test sets.

    This class encapsulates the entire preprocessing pipeline from the provided
    Jupyter Notebook. It loads raw data, processes it, and saves the final
    datasets as pickle files.
    """

    def __init__(self, users_path, categories_path, dictionary_path, unigrams_path, output_dir, test_size=0.2,
                 random_state=42):
        """
        Initializes the UserDataPreprocessor with necessary file paths and parameters.

        Args:
            users_path (str): Path to the 'jobs-users' file.
            categories_path (str): Path to the 'major_categories' file.
            dictionary_path (str): Path to the 'dictionary' file.
            unigrams_path (str): Path to the 'jobs-unigrams' file.
            output_dir (str): Directory to save the processed .pkl files.
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int): Seed for the random number generator for reproducibility.
        """
        self.users_path = users_path
        self.categories_path = categories_path
        self.dictionary_path = dictionary_path
        self.unigrams_path = unigrams_path
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state

        self.users_df = None
        self.word_dictionary = {}
        self.user_unigrams = {}
        self.stopwords = None

        # Initialize tqdm for pandas
        tqdm.pandas(desc="Processing Users")

    def _load_and_process_users(self):
        """
        Loads user data, extracts major occupation categories, and merges
        it with category names. This corresponds to Section 1 of the notebook.
        """
        print("1. Loading and processing user and occupation data...")
        # Load users and extract the major occupation category
        self.users_df = pd.read_csv(self.users_path, sep=' ', header=None, names=['user_id', 'occupation_code'])
        self.users_df['occupation_code'] = self.users_df['occupation_code'].astype(str).str[0].astype(int)

        # Load category names and merge with the users DataFrame
        major_categories_df = pd.read_csv(self.categories_path, sep=':', header=None,
                                          names=['occupation_code', 'category'])
        self.users_df = pd.merge(self.users_df, major_categories_df, on='occupation_code', how='left')

        print(f"Successfully loaded and processed {len(self.users_df)} users.")
        print("Value counts for each major group:\n", self.users_df['occupation_code'].value_counts())
        print("-" * 30)

    def _load_text_data(self):
        """
        Loads the word dictionary and user unigrams from their respective files.
        This corresponds to the start of Section 2 of the notebook.
        """
        print("2. Loading text data (dictionary and unigrams)...")
        # Load word dictionary
        print(f"Loading dictionary from {self.dictionary_path}...")
        with open(self.dictionary_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.word_dictionary[parts[0]] = parts[1]
        print(f"Dictionary loaded with {len(self.word_dictionary)} words.")

        # Load user unigrams
        print(f"Loading user unigrams from {self.unigrams_path}...")
        with open(self.unigrams_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user_id = parts[0]
                    features_str = ' '.join(parts[1:])
                    self.user_unigrams[user_id] = features_str
        print(f"Unigrams loaded for {len(self.user_unigrams)} users.")
        print("-" * 30)

    def _get_user_word_array(self, user_id):
        """
        Creates an aggregated list of words for a single user based on unigram frequency,
        filtering for stopwords, length, and non-alphabetic characters.
        """
        user_id_str = str(user_id)
        if user_id_str not in self.user_unigrams:
            return []

        features_str = self.user_unigrams[user_id_str]
        word_tokens = features_str.split()
        aggregated_words = []

        for token in word_tokens:
            try:
                word_id, frequency_str = token.split(':')
                frequency = int(frequency_str)
                if word_id in self.word_dictionary:
                    word = self.word_dictionary[word_id]
                    if word.isalpha() and len(word) > 2 and word not in self.stopwords:
                        aggregated_words.extend([word] * frequency)
            except ValueError:
                continue
        return aggregated_words

    def _process_and_aggregate_words(self):
        """
        Applies the word aggregation function to each user in the DataFrame,
        creating a new column with the list of processed words.
        """
        print("3. Processing unigrams and creating the aggregated word column...")
        # Download and load NLTK stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            print("NLTK stopwords not found. Downloading...")
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))
        print(f"Loaded {len(self.stopwords)} stopwords from NLTK.")

        # Apply the processing function to the 'user_id' column
        self.users_df['aggregated_words'] = self.users_df['user_id'].progress_apply(self._get_user_word_array)

        print("Processing complete.")

        # Verification for a sample user
        example_user_id = 206749819
        if not self.users_df[self.users_df['user_id'] == example_user_id].empty:
            example_words = self.users_df[self.users_df['user_id'] == example_user_id]['aggregated_words'].values[0]
            print(f"\n--- Verification for user {example_user_id} ---")
            print(f"Total words: {len(example_words)}")
            print(f"Unique words: {len(set(example_words))}")
        print("-" * 30)

    def _split_and_save_data(self):
        """
        Splits the preprocessed data into training and test sets and saves them
        as pickle files. This corresponds to Section 3 and the final part of the notebook.
        """
        print("4. Splitting the dataset and saving to files...")
        y = self.users_df['occupation_code']

        # Stratified split to maintain class distribution
        train_df, test_df = train_test_split(
            self.users_df, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print("\nTraining set class distribution:\n", train_df['occupation_code'].value_counts())
        print("\nTest set class distribution:\n", test_df['occupation_code'].value_counts())
        print(f"\nFinal training set size: {train_df.shape}")
        print(f"Final test set size: {test_df.shape}")

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"\nCreated output directory: {self.output_dir}")

        # Define file paths and save datasets
        train_path = os.path.join(self.output_dir, 'train.pkl')
        test_path = os.path.join(self.output_dir, 'test.pkl')

        train_df.to_pickle(train_path)
        print(f"Training set saved to {train_path}")

        test_df.to_pickle(test_path)
        print(f"Test set saved to {test_path}")

        print("\nAll files saved successfully.")
        print("-" * 30)

    def process(self):
        """
        Executes the entire preprocessing pipeline in the correct order.
        """
        self._load_and_process_users()
        self._load_text_data()
        self._process_and_aggregate_words()
        self._split_and_save_data()
        print("\nPreprocessing pipeline finished.")


if __name__ == '__main__':
    # --- Configuration ---
    # Define base paths relative to the script location
    # IMPORTANT: Adjust these paths to match your directory structure.
    BASE_DATA_DIR = '../data'
    RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'processed')

    # Define paths to the raw data files
    users_file_path = os.path.join(RAW_DATA_DIR, 'jobs-users')
    categories_file_path = os.path.join(RAW_DATA_DIR, 'major_categories')
    dictionary_file_path = os.path.join(RAW_DATA_DIR, 'dictionary')
    unigrams_file_path = os.path.join(RAW_DATA_DIR, 'jobs-unigrams')

    # --- Execution ---
    # Check if raw data files exist before running
    required_files = [users_file_path, categories_file_path, dictionary_file_path, unigrams_file_path]
    if not all(os.path.exists(f) for f in required_files):
        print(
            "Error: One or more raw data files not found. Please check the paths in the `if __name__ == '__main__':` block.")
    else:
        # Instantiate the preprocessor
        preprocessor = Processor(
            users_path=users_file_path,
            categories_path=categories_file_path,
            dictionary_path=dictionary_file_path,
            unigrams_path=unigrams_file_path,
            output_dir=PROCESSED_DATA_DIR
        )

        # Run the entire preprocessing pipeline
        preprocessor.process()