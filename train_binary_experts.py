"""
🚀 MEGA-COUNCIL TRAINER
Bootstraps 12+ experts to rival Google's metadata categorization.
Uses 20 Newsgroups (Real Data) + Weighted Keyword Heuristics.
"""
import logging
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from app.experts import council

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("trainer")

# =========================================================
# HELPER: Keyword Booster
# =========================================================
def train_from_keywords(expert_name, positive_phrases, negative_phrases, weight=30):
    """
    Trains an expert using weighted keyword lists.
    We repeat the lists 'weight' times to simulate a large dataset.
    """
    logger.info(f"⚡ Training '{expert_name}' (Heuristic Mode)...")
    
    # Repeat data to establish strong priors
    X = (positive_phrases * weight) + (negative_phrases * weight)
    y = ([1] * len(positive_phrases) * weight) + ([0] * len(negative_phrases) * weight)
    
    council.train(expert_name, X, y)

# =========================================================
# 1. REAL DATA EXPERTS (Using Scikit-Learn)
# =========================================================
def train_real_data_experts():
    logger.info("📥 Downloading 20 Newsgroups Data (this may take a minute)...")
    
    # --- A. POLITICS EXPERT ---
    # Real discussions on guns, middle-east, and misc politics
    politics = fetch_20newsgroups(subset='train', categories=['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'], remove=('headers', 'footers', 'quotes'))
    misc = fetch_20newsgroups(subset='train', categories=['comp.graphics', 'rec.sport.baseball'], remove=('headers', 'footers', 'quotes'))
    council.train("is_politics", politics.data + misc.data, [1]*len(politics.data) + [0]*len(misc.data))
    
    # --- B. SPORTS EXPERT ---
    # Real discussions on baseball and hockey
    sports = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball', 'rec.sport.hockey'], remove=('headers', 'footers', 'quotes'))
    council.train("is_sports", sports.data + politics.data, [1]*len(sports.data) + [0]*len(politics.data))
    
    # --- C. TECH EXPERT (The original) ---
    tech = fetch_20newsgroups(subset='train', categories=['comp.sys.mac.hardware', 'comp.windows.x', 'sci.crypt'], remove=('headers', 'footers', 'quotes'))
    council.train("is_tech", tech.data + sports.data, [1]*len(tech.data) + [0]*len(sports.data))

    # --- D. SCIENCE/MEDICINE EXPERT ---
    med = fetch_20newsgroups(subset='train', categories=['sci.med', 'sci.space'], remove=('headers', 'footers', 'quotes'))
    council.train("is_science", med.data + politics.data, [1]*len(med.data) + [0]*len(politics.data))

# =========================================================
# 2. FORMAT EXPERTS (What kind of page is this?)
# =========================================================
def train_format_experts():
    # --- IS_FORUM ---
    # Detects Reddit, StackOverflow, Bulletin Boards
    forum_pos = [
        "thread closed", "sticky post", "reply to this thread", "original poster", "OP",
        "joined date", "total posts", "reputation score", "upvote", "downvote",
        "moderator", "senior member", "quote from", "last edited by", "view profile"
    ]
    forum_neg = ["breaking news", "abstract", "introduction", "bibliography", "terms of service", "add to cart"]
    train_from_keywords("is_forum", forum_pos, forum_neg)

    # --- IS_ACADEMIC ---
    # Detects Papers, Journals, Documentation
    academic_pos = [
        "abstract", "introduction", "methodology", "conclusion", "references",
        "bibliography", "cited by", "doi:", "figure 1", "table 2", "et al.",
        "university press", "peer reviewed", "journal of", "thesis", "dissertation"
    ]
    academic_neg = ["buy now", "click here", "subscribe", "hot singles", "top 10 list", "lol", "lmao"]
    train_from_keywords("is_academic", academic_pos, academic_neg)

    # --- IS_VIDEO ---
    # Detects YouTube, Vimeo, Twitch pages (even if just metadata)
    video_pos = [
        "duration", "views", "subscribe", "watch later", "full screen", 
        "autoplay", "live stream", "4k", "1080p", "transcript", "cast to device",
        "uploaded by", "streamer", "vlog"
    ]
    train_from_keywords("is_video", video_pos, academic_neg)

# =========================================================
# 3. TOPIC EXPERTS (Heuristic)
# =========================================================
def train_topic_experts():
    # --- IS_FINANCE ---
    finance_pos = [
        "stock market", "nasdaq", "dow jones", "cryptocurrency", "bitcoin", "ethereum",
        "inflation rate", "quarterly earnings", "dividend yield", "market cap",
        "bull market", "bear market", "portfolio", "investment strategy", "capital gains",
        "federal reserve", "interest rates", "mortgage", "401k", "roth ira"
    ]
    finance_neg = ["recipe", "sports score", "movie review", "fashion trends", "celebrity gossip"]
    train_from_keywords("is_finance", finance_pos, finance_neg)

    # --- IS_HEALTH ---
    health_pos = [
        "symptoms", "diagnosis", "treatment", "side effects", "prescription",
        "consult your doctor", "calories", "diet", "nutrition", "workout routine",
        "mental health", "therapy", "vaccine", "epidemic", "immune system",
        "blood pressure", "heart rate", "supplements", "vitamins"
    ]
    train_from_keywords("is_health", health_pos, finance_neg)

    # --- IS_TRAVEL ---
    travel_pos = [
        "best hotels", "flight booking", "itinerary", "things to do in", "tour guide",
        "vacation rental", "airbnb", "passport", "visa requirements", "backpacking",
        "resort", "beaches", "hiking trails", "national park", "travel tips"
    ]
    train_from_keywords("is_travel", travel_pos, finance_neg)

# =========================================================
# 4. TONE/QUALITY EXPERTS
# =========================================================
def train_tone_experts():
    # --- IS_OPINION ---
    # Detects subjective, emotional, or argumentative content
    opinion_pos = [
        "in my opinion", "i believe", "disgraceful", "amazing", "worst ever",
        "unbelievable", "should be", "must stop", "i feel", "personally",
        "rant", "review", "verdict", "pros and cons", "why i hate", "why i love"
    ]
    opinion_neg = [
        "according to data", "the study found", "reportedly", "stated", "observed",
        "measured", "calculated", "evidence suggests", "abstract", "methodology"
    ]
    train_from_keywords("is_opinion", opinion_pos, opinion_neg)

    # --- IS_SPAM (Refined) ---
    spam_pos = [
        "buy cheap", "guaranteed winner", "click here to claim", "viagra", "cialis",
        "meet women", "hot singles", "crypto giveaway", "urgent wire transfer",
        "verify account now", "password expired", "risk free", "double your money"
    ]
    train_from_keywords("is_spam", spam_pos, ["history of", "tutorial", "recipe"])

    # --- IS_COMMERCIAL (Refined) ---
    comm_pos = [
        "add to cart", "checkout", "price", "discount", "sale", "limited time",
        "subscription", "pricing", "buy now", "order online", "shipping"
    ]
    train_from_keywords("is_commercial", comm_pos, ["history", "theory", "abstract"])


if __name__ == "__main__":
    print("🧠 Spinning up the Mega-Council...")
    
    try:
        # 1. The Heavy Lifters (Real Data)
        train_real_data_experts()
        
        # 2. The Specialists (Keywords)
        train_format_experts()
        train_topic_experts()
        train_tone_experts()
        
        print("\n✅ MEGA-COUNCIL READY: 12 Experts Active.")
        print("   - Topics: Tech, Politics, Sports, Science, Finance, Health, Travel")
        print("   - Formats: Forum, Academic, Video")
        print("   - Tone: Opinion, Commercial, Spam")
        
        # Validation
        print("\n🧪 VALIDATION TEST:")
        test = "The study found that inflation affects stock market returns. See Figure 1."
        print(f"   Input: '{test}'")
        res = council.analyze(test)
        
        # Filter for display (show only confidence > 50%)
        active = {k: f"{v:.2f}" for k,v in res.items() if v > 0.5}
        print(f"   Result: {active}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
