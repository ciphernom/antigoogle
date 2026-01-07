import os
import pickle
import logging
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

logger = logging.getLogger(__name__)

# Ensure this path matches your docker-compose volume mapping
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "experts.pkl")

class BinaryExpert:
    """A single specialist that only answers Yes/No for its specific field."""
    def __init__(self, name):
        self.name = name
        self.model = MultinomialNB(alpha=0.01)
        self.is_trained = False
        # We enforce strict binary classes: [0=No, 1=Yes]
        self.classes = [0, 1] 

    def train(self, X, y):
        """
        y must be a list of 0s (No) and 1s (Yes)
        """
        self.model.partial_fit(X, y, classes=self.classes)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            return 0.5 # Unsure
        # Return probability of "Yes" (class 1)
        return self.model.predict_proba(X)[0][1]

class CouncilOfExperts:
    def __init__(self):
        # FIX 1: Ensure directory exists so we don't crash on save
        os.makedirs(DATA_DIR, exist_ok=True)

        # FIX 2: Naive Bayes works best with raw counts, not normalized floats.
        # alternate_sign=False is REQUIRED for MultinomialNB (cannot handle negative inputs).
        self.vectorizer = HashingVectorizer(
            n_features=2**18, 
            alternate_sign=False, 
            norm=None  # <--- Added: Keep raw counts for better NB accuracy
        )
        
        self.experts = {} # Dictionary of BinaryExpert objects
        self.load()

    def get_expert(self, name):
        """Get or create a new expert on the fly."""
        if name not in self.experts:
            self.experts[name] = BinaryExpert(name)
        return self.experts[name]

    def train(self, expert_name, texts, is_match):
        """
        expert_name: e.g., 'is_tech', 'is_spam'
        texts: list of strings
        is_match: list of booleans or ints (1=Yes, 0=No)
        """
        expert = self.get_expert(expert_name)
        X = self.vectorizer.transform(texts)
        
        # Ensure labels are 0 or 1
        y = [1 if label else 0 for label in is_match]
        
        expert.train(X, y)
        self.save()
        logger.info(f"🎓 Trained expert '{expert_name}' on {len(texts)} samples.")

    def analyze(self, text):
        """Ask EVERY expert for their binary opinion."""
        results = {}
        if not text or len(text) < 50:
            return results

        X = self.vectorizer.transform([text])
        
        for name, expert in self.experts.items():
            if expert.is_trained:
                # Get probability (0.0 to 1.0)
                score = expert.predict(X)
                
                # Only report if confidence is somewhat high to reduce noise?
                # Currently we return everything, which is fine for debugging.
                results[name] = float(score)
                
        return results

    def save(self):
        try:
            with open(DATA_PATH, "wb") as f:
                pickle.dump(self.experts, f)
        except Exception as e:
            logger.error(f"Failed to save experts: {e}")

    def load(self):
        if os.path.exists(DATA_PATH):
            try:
                with open(DATA_PATH, "rb") as f:
                    self.experts = pickle.load(f)
                logger.info(f"🧠 Council loaded. Active experts: {list(self.experts.keys())}")
            except Exception as e:
                logger.warning(f"Could not load experts: {e}")

# Global instance
council = CouncilOfExperts()
