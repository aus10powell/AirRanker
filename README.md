# AirRanker

## **Objective**
Develop a recommendation system for Airbnb listings that can be expanded to different regions. The system will leverage listing metadata, user reviews, and embeddings to rank recommendations using **StarRanker**.

---

## **Phases of the Project**

### **1. Data Collection & Preprocessing**
**Goal:** Prepare raw Airbnb datasets for recommendation.  
#### **Tasks:**
- ✅ Load Airbnb dataset (listings, reviews, calendar availability).
- ✅ Extract relevant features:
  - **Listings:** Price, amenities, location (lat/lon), room type, property type.
  - **Reviews:** Sentiment, text embeddings, booking history.
  - **User Interaction Data (if available):** Bookings, search history.
- ✅ Preprocess text fields:
  - Remove **HTML tags**, stopwords, lemmatization.
  - Convert **review text to embeddings** (e.g., OpenAI, SBERT).
- ✅ **Create holdout dataset:**
  - Exclude booked listings for validation.
  - Reserve a portion of **past bookings** for evaluation.

---

### **2. Building the Recommendation System**
**Goal:** Implement **StarRanker-based** personalized listing ranking.  
#### **Tasks:**
- ✅ **Generate embeddings** for listings using text, amenities, and location data.
- ✅ Implement **pairwise ranking (StarRanker) for zero-shot ranking** of listings.
- ✅ Ensure model can be **generalized to different locations** (Seattle first).
- ✅ Implement **filters** (price range, property type, availability).
- ✅ Store **precomputed embeddings** for fast retrieval.

---

### **3. Validation & Metrics**
**Goal:** Evaluate ranking quality against **actual user choices**.  
#### **Tasks:**
- ✅ **Hold-out validation:**
  - Check if booked listings appear in top recommendations.
  - Compare model predictions with actual booking behavior.
- ✅ **Ranking Metrics:**
  - **MRR (Mean Reciprocal Rank):** Measures if correct listing appears early.
  - **Hit@K:** Checks if true booking appears in top-K results.
  - **NDCG (Normalized Discounted Cumulative Gain):** Measures ranking quality.
- ✅ **Sanity Checks:**
  - Does the model **preferably rank higher-rated listings**?
  - Does ranking **change meaningfully with different price ranges**?
- ✅ **Baseline Comparison:**
  - Compare **StarRanker vs. a naive popularity-based method** (e.g., most-reviewed listings).

---

### **4. Streamlit Integration (Future Phase)**
**Goal:** Build an interactive dashboard for recommendations.  
#### **Tasks:**
- ✅ Implement a **Streamlit app** with:
  - Search by **city/region (Seattle first, later extendable)**.
  - Compare **StarRanker vs. Popularity ranking**.
  - Filter by **price, property type, availability**.
- ✅ Visualize ranking performance:
  - Show **ranked listings with review snippets**.
  - Display **evaluation metrics (MRR, Hit@K, etc.)**.
- ✅ Allow users to test different ranking methods.

---

## **Next Steps**
- ✅ **Implement preprocessing pipeline (feature extraction, embeddings, holdout creation).**
- ✅ **Build ranking function & validation metrics (MRR, Hit@K, NDCG).**
- ✅ **Run baseline comparison & sanity checks.**
- ✅ **Start Streamlit integration for visualization.**


### Project Structure

airbnb_recommender/
│
├── data/  
│   ├── raw/                      # Raw Airbnb dataset files  
│   ├── processed/                 # Preprocessed data & embeddings  
│   ├── holdout_bookings.csv       # Ground truth for validation  
│   ├── Seattle_config.yaml        # Config for Seattle-specific settings  
│   ├── general_config.yaml        # General settings (e.g., embedding model)  
│
├── src/  
│   ├── preprocessing/  
│   │   ├── preprocess_listings.py  # Cleans and extracts listing features  
│   │   ├── preprocess_reviews.py   # Processes review text into embeddings  
│   │   ├── data_loader.py          # Loads data from CSV, Parquet, etc.  
│   │  
│   ├── recommendation/  
│   │   ├── embeddings.py           # Generates embeddings for listings  
│   │   ├── ranking.py              # Implements StarRanker recommendation logic  
│   │   ├── recommend.py            # Main module to generate recommendations  
│   │  
│   ├── evaluation/  
│   │   ├── validation.py           # Computes MRR, Hit@K, NDCG  
│   │   ├── sanity_checks.py        # Performs sanity checks on dataset  
│   │  
│   ├── streamlit_app/  
│   │   ├── app.py                  # Streamlit interface for testing recommendations  
│   │   ├── visualization.py         # Plots ranking comparisons  
│
├── tests/  
│   ├── test_preprocessing.py       # Unit tests for data processing  
│   ├── test_recommendation.py      # Tests for embeddings & ranking  
│   ├── test_validation.py          # Tests for evaluation metrics  
│
├── notebooks/                      # Jupyter notebooks for exploratory analysis  
│
├── requirements.txt                 # Dependencies  
├── README.md                        # Project overview  
└── .gitignore                        # Ignore unnecessary files  


## Lessons Learned


1. **Memory Management Proved Essential**: We discovered that memory efficiency was crucial when working with large datasets. Instead of pre-computing entire similarity matrices, we implemented on-demand computation and utilized sparse matrices which dramatically reduced our memory footprint. Adding manual garbage collection between batches further optimized performance.

2. **Hybrid Recommendation Approach Yielded Better Results**: By combining semantic similarity (based on listing content) with collaborative filtering (based on user behavior), we created more robust recommendations that captured both content relevance and user preference patterns.

3. **Smaller Models Delivered Sufficient Performance**: We found that using smaller transformer models (MiniLM) with reduced batch sizes provided an excellent balance between accuracy and resource consumption, allowing our recommendation system to run efficiently even with limited computing resources.