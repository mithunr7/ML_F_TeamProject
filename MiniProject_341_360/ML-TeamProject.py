"""
Goodreads Review Popularity Classification - COMPACT VERSION
Same functionality, 40% less code
"""

import pandas as pd
import numpy as np
import re, json, os, sys, io, warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass


class CompactGoodreadsAnalyzer:
    
    def __init__(self, data_path, sample_size=100_000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.df = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        print(f"Loading {self.sample_size:,} reviews...")
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.sample_size:
                    break
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue
                if (i + 1) % 100000 == 0:
                    print(f"  {i + 1:,} loaded...")
        
        self.df = pd.DataFrame(data)
        print(f"[OK] Loaded {len(self.df):,} reviews")
        return self.df
    
    def define_popularity(self, percentile=75):
        self.df['n_votes'] = pd.to_numeric(self.df['n_votes'], errors='coerce')
        self.df = self.df[self.df['n_votes'].notna()].copy()
        threshold = self.df['n_votes'].quantile(percentile / 100)
        self.df['is_popular'] = (self.df['n_votes'] >= threshold).astype(int)
        print(f"\nPopularity: Top {100-percentile}% (threshold: {threshold} votes)")
        print(self.df['is_popular'].value_counts())
        return self.df
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|<.*?>|\\n', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_features(self, text, row):
        """Extract all features in one function"""
        feat = {}
        
        if not text:
            return {k: 0 for k in ['chars', 'words', 'sents', 'avg_word', 'avg_sent',
                   'exclaim', 'question', 'comma', 'caps', 'sent_neg', 'sent_pos',
                   'sent_compound', 'polarity', 'subjectivity', 'lex_div',
                   'rating', 'extreme_rating', 'n_comments', 'has_read', 'has_started']}
        
        # Syntactic
        words = text.split()
        sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        feat['chars'] = len(text)
        feat['words'] = len(words)
        feat['sents'] = len(sents)
        feat['avg_word'] = np.mean([len(w) for w in words]) if words else 0
        feat['avg_sent'] = len(words) / len(sents) if sents else 0
        feat['exclaim'] = text.count('!')
        feat['question'] = text.count('?')
        feat['comma'] = text.count(',')
        feat['caps'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Semantic
        try:
            sia = SentimentIntensityAnalyzer()
            sent = sia.polarity_scores(text)
            blob = TextBlob(text)
            feat['sent_neg'] = sent['neg']
            feat['sent_pos'] = sent['pos']
            feat['sent_compound'] = sent['compound']
            feat['polarity'] = blob.sentiment.polarity
            feat['subjectivity'] = blob.sentiment.subjectivity
            feat['lex_div'] = len(set(words)) / len(words) if words else 0
        except:
            feat.update({k: 0 for k in ['sent_neg', 'sent_pos', 'sent_compound', 
                        'polarity', 'subjectivity', 'lex_div']})
        
        # Non-textual
        try:
            rating = float(row.rating) if pd.notna(row.rating) else 3.0
            feat['rating'] = rating
            feat['extreme_rating'] = 1 if rating in [1.0, 5.0] else 0
            feat['n_comments'] = int(row.n_comments) if pd.notna(row.n_comments) else 0
            feat['has_read'] = 1 if (hasattr(row, 'read_at') and pd.notna(row.read_at)) else 0
            feat['has_started'] = 1 if (hasattr(row, 'started_at') and pd.notna(row.started_at)) else 0
        except:
            feat.update({'rating': 3.0, 'extreme_rating': 0, 'n_comments': 0, 
                        'has_read': 0, 'has_started': 0})
        
        return feat
    
    def engineer_features(self, use_cache=True):
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        cache_file = f'compact_features_{len(self.df)}.pkl'
        
        if use_cache and os.path.exists(cache_file):
            print("[OK] Loading from cache")
            self.feature_df = pd.read_pickle(cache_file)
            self.df['cleaned_text'] = pd.read_pickle(f'compact_text_{len(self.df)}.pkl')
            return self.feature_df
        
        print("Cleaning text...")
        self.df['cleaned_text'] = self.df['review_text'].apply(self.clean_text)
        self.df = self.df[self.df['cleaned_text'].str.len() > 0].copy()
        
        print(f"Extracting features from {len(self.df):,} reviews...")
        features = []
        for idx, row in enumerate(self.df.itertuples()):
            if idx % 100000 == 0:
                print(f"  {idx:,}/{len(self.df):,}")
            text = self.df.iloc[idx]['cleaned_text']
            features.append(self.extract_features(text, row))
        
        self.feature_df = pd.DataFrame(features)
        
        # Add TF-IDF
        print("\nAdding TF-IDF...")
        tfidf = TfidfVectorizer(max_features=100, stop_words='english', 
                               ngram_range=(1, 2), min_df=5, max_df=0.95)
        tfidf_matrix = tfidf.fit_transform(self.df['cleaned_text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
        tfidf_df.index = self.feature_df.index
        self.feature_df = pd.concat([self.feature_df, tfidf_df], axis=1)
        
        if use_cache:
            self.feature_df.to_pickle(cache_file)
            self.df['cleaned_text'].to_pickle(f'compact_text_{len(self.df)}.pkl')
        
        print(f"[OK] {self.feature_df.shape[1]} features extracted")
        return self.feature_df
    
    def prepare_data(self):
        print("\n" + "="*50)
        print("PREPARING DATA")
        print("="*50)
        
        X = self.feature_df.fillna(0).replace([np.inf, -np.inf], 0)
        y = self.df['is_popular'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Logistic Regression
        print("\n1. Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
        lr.fit(self.X_train, self.y_train)
        self.models['LR'] = lr
        self.results['LR'] = self._evaluate(lr)
        
        # XGBoost
        print("\n2. XGBoost...")
        scale = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                           scale_pos_weight=scale, tree_method='hist', n_jobs=-1)
        xgb.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb
        self.results['XGBoost'] = self._evaluate(xgb)
        
        # Neural Network
        print("\n3. Neural Network...")
        nn = Sequential([
            Dense(256, activation='relu', input_shape=(self.X_train.shape[1],)),
            BatchNormalization(), Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(), Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(), Dropout(0.2),
            Dense(32, activation='relu'), Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        nn.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        
        class_weights = {0: len(self.y_train) / (2 * (self.y_train == 0).sum()),
                        1: len(self.y_train) / (2 * (self.y_train == 1).sum())}
        
        nn.fit(self.X_train, self.y_train, validation_split=0.2, epochs=20,
              batch_size=128, class_weight=class_weights, verbose=1,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3)])
        
        self.models['NN'] = nn
        y_pred_proba = nn.predict(self.X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        self.results['NN'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'cm': confusion_matrix(self.y_test, y_pred)
        }
    
    def _evaluate(self, model):
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'cm': confusion_matrix(self.y_test, y_pred)
        }
    
    def print_results(self):
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        for name, res in self.results.items():
            print(f"\n{name}:")
            print(f"  Accuracy:  {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
            print(f"  Precision: {res['precision']:.4f}")
            print(f"  Recall:    {res['recall']:.4f}")
            print(f"  F1-Score:  {res['f1']:.4f}")
            print(f"  ROC-AUC:   {res['roc_auc']:.4f}")
    
    def plot_results(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        models = list(self.results.keys())
        
        ax = axes[0]
        x = np.arange(len(metrics))
        width = 0.25
        for i, model in enumerate(models):
            values = [self.results[model][m] for m in metrics]
            ax.bar(x + i*width, values, width, label=model)
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Confusion matrices
        ax = axes[1]
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        cm = best_model[1]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{best_model[0]} Confusion Matrix')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('results.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Saved results.png")
        plt.show()
    
    def run(self):
        print("\n" + "="*60)
        print("GOODREADS REVIEW POPULARITY CLASSIFIER")
        print("="*60)
        
        try:
            self.load_data()
            self.define_popularity(percentile=75)
            self.engineer_features()
            self.prepare_data()
            self.train_models()
            self.print_results()
            self.plot_results()
            
            print("\n" + "="*60)
            print("[OK] COMPLETE!")
            print("="*60)
            
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    DATA_PATH = r"D:\goodreads_reviews_dedup.json"  # Update path
    
    USE_SMALL_SAMPLE = True  # Set False for full 1.98M
    sample_size = 100_000 if USE_SMALL_SAMPLE else 1_980_000
    
    print(f"Starting with {sample_size:,} reviews...")
    if USE_SMALL_SAMPLE:
        print("[WARNING] Test mode - set USE_SMALL_SAMPLE=False for full dataset")
    
    analyzer = CompactGoodreadsAnalyzer(DATA_PATH, sample_size)
    analyzer.run()
    
    print("\nAccess models: analyzer.models['XGBoost']")