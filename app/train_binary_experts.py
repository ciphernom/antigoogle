"""
🚀 MEGA-COUNCIL TRAINER (Direct Loader)
Uses `load_files` to directly read the folders you extracted.
"""
import logging
import os
from sklearn.datasets import load_files
from app.experts import council

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("trainer")

# Points directly to where you extracted the train folder
# Structure: /app/data/20news_home/20news-bydate-train/comp.graphics/...
DATA_HOME = "/app/data/20news_home/20news-bydate-train"

def train_from_keywords(expert_name, positive_phrases, negative_phrases, weight=30):
    logger.info(f"⚡ Training '{expert_name}' (Heuristic Mode)...")
    X = (positive_phrases * weight) + (negative_phrases * weight)
    y = ([1] * len(positive_phrases) * weight) + ([0] * len(negative_phrases) * weight)
    council.train(expert_name, X, y)

def load_category_data(categories):
    """
    Directly loads text files from the disk.
    Strips email headers to ensure we train on CONTENT, not metadata.
    """
    try:
        # load_files reads the raw text from the folders
        dataset = load_files(
            DATA_HOME, 
            categories=categories, 
            encoding='latin1', # The dataset uses legacy encoding
            shuffle=True, 
            random_state=42
        )
        
        # Strip Headers (Everything before the first blank line)
        # This prevents the model from cheating by looking at "Subject:" lines
        clean_data = []
        for text in dataset.data:
            parts = text.split('\n\n', 1)
            if len(parts) > 1:
                clean_data.append(parts[1])
            else:
                clean_data.append(text)
                
        return clean_data
    except FileNotFoundError:
        raise OSError(f"Folder not found: {DATA_HOME}")

def train_real_data_experts():
    logger.info(f"📂 Reading raw text files from: {DATA_HOME}")
    
    try:
        # --- A. POLITICS ---
        pol_cats = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc']
        misc_cats = ['comp.graphics', 'rec.sport.baseball']
        
        pol_data = load_category_data(pol_cats)
        misc_data = load_category_data(misc_cats)
        
        council.train("is_politics", pol_data + misc_data, [1]*len(pol_data) + [0]*len(misc_data))
        
        # --- B. SPORTS ---
        sport_cats = ['rec.sport.baseball', 'rec.sport.hockey']
        pol_ref = ['talk.politics.guns'] # Use politics as contrast
        
        sport_data = load_category_data(sport_cats)
        contrast_data = load_category_data(pol_ref)
        
        council.train("is_sports", sport_data + contrast_data, [1]*len(sport_data) + [0]*len(contrast_data))
        
        # --- C. TECH ---
        tech_cats = ['comp.sys.mac.hardware', 'comp.windows.x', 'sci.crypt']
        tech_data = load_category_data(tech_cats)
        
        # Use sports as contrast for tech
        council.train("is_tech", tech_data + sport_data, [1]*len(tech_data) + [0]*len(sport_data))

        # --- D. SCIENCE ---
        sci_cats = ['sci.med', 'sci.space']
        sci_data = load_category_data(sci_cats)
        
        council.train("is_science", sci_data + sport_data, [1]*len(sci_data) + [0]*len(sport_data))
        
        logger.info("✅ Real Data Experts Trained Successfully.")

    except Exception as e:
        logger.error(f"❌ Failed to load local data: {e}")
        logger.warning(f"Ensure path exists: {DATA_HOME}")
        train_real_data_fallback()

def train_real_data_fallback():
    logger.warning("⚠️ Using FALLBACK KEYWORDS for heavy lifters.")
    train_from_keywords("is_tech", ["software", "hardware", "cpu", "linux", "python", "windows"], ["sports", "recipe"])
    train_from_keywords("is_politics", ["government", "election", "president", "congress", "vote"], ["java", "vacation"])
    train_from_keywords("is_sports", ["game", "team", "score", "player", "league", "win"], ["server", "finance"])
    train_from_keywords("is_science", ["research", "study", "experiment", "lab", "biology", "physics"], ["news", "opinion"])

def train_standard_experts():
    # Formats
    train_from_keywords("is_forum", ["thread", "post", "op", "upvote", "forum"], ["article", "shop"])
    train_from_keywords("is_academic", ["abstract", "references", "doi", "journal"], ["buy", "blog"])
    train_from_keywords("is_video", ["duration", "views", "subscribe", "1080p"], ["text"])
    # Topics
    train_from_keywords("is_finance", ["stock", "market", "crypto", "bitcoin", "investment"], ["game", "food"])
    train_from_keywords("is_health", ["doctor", "symptoms", "treatment", "medicine"], ["computer", "finance"])
    train_from_keywords("is_travel", ["hotel", "flight", "booking", "resort", "trip"], ["server", "linux"])
    # Tone
    train_from_keywords("is_opinion", ["i believe", "in my opinion", "review", "worst"], ["data", "report"])
    train_from_keywords("is_spam", ["buy now", "winner", "viagra", "crypto giveaway"], ["tutorial", "news"])
    train_from_keywords("is_commercial", ["cart", "checkout", "price", "sale"], ["history", "wiki"])

if __name__ == "__main__":
    print("🧠 Spinning up the Mega-Council (Direct Load Mode)...")
    if os.path.exists(DATA_HOME):
        train_real_data_experts()
    else:
        print(f"❌ Path not found: {DATA_HOME}")
        train_real_data_fallback()
        
    train_standard_experts()
    print("\n✅ MEGA-COUNCIL READY.")
