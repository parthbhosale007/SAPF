# crop_doctor_boosted.py
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------
# 1️⃣ Load Dataset
# ---------------------------
df = pd.read_csv("Crop_Doctor/crop_diseases.csv")
df['Region'] = df['Region'].fillna('All India')

print(f"📊 Dataset loaded: {len(df)} diseases across {df['Crop'].nunique()} crops")

# ---------------------------
# 2️⃣ ENHANCED Text Cleaning with Symptom Keywords
# ---------------------------
def enhanced_clean_symptoms(text):
    text = str(text).lower()
    
    # Keep important disease-related punctuation
    text = re.sub(r'[^a-z\s.,;-]', ' ', text)
    
    # Expand common abbreviations and synonyms
    synonym_map = {
        'spots': 'spot lesion',
        'lesions': 'lesion spot',
        'yellowing': 'yellow',
        'browning': 'brown', 
        'wilting': 'wilt',
        'rotting': 'rot',
        'blight': 'blight spot',
        'mildew': 'mildew powder',
        'pustules': 'pustule spot',
        'blight': 'blight disease',
        'rot': 'rot disease'
    }
    
    for word, replacement in synonym_map.items():
        text = text.replace(word, replacement)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Symptoms_enhanced'] = df['Symptoms'].apply(enhanced_clean_symptoms)

# ---------------------------
# 3️⃣ CREATE RICHER SEARCH TEXT with weighted terms
# ---------------------------
def create_search_text(row):
    crop = row['Crop'].lower()
    symptoms = row['Symptoms_enhanced']
    region = row['Region'].lower()
    
    # Weight important terms by repeating them
    critical_terms = []
    
    # Add disease type emphasis
    if 'fungal' in row['Cause'].lower():
        critical_terms.extend(['fungal', 'fungus'] * 2)
    if 'bacterial' in row['Cause'].lower():
        critical_terms.extend(['bacterial', 'bacteria'] * 2)
    if 'viral' in row['Cause'].lower():
        critical_terms.extend(['viral', 'virus'] * 2)
    
    # Add season emphasis
    if 'kharif' in row['Season'].lower():
        critical_terms.extend(['kharif', 'monsoon'] * 2)
    if 'rabi' in row['Season'].lower():
        critical_terms.extend(['rabi', 'winter'] * 2)
    
    search_text = f"{crop} {symptoms} {region} {' '.join(critical_terms)}"
    return search_text

df['Search_Text_Enhanced'] = df.apply(create_search_text, axis=1)

# ---------------------------
# 4️⃣ IMPROVED VECTORIZER with better parameters
# ---------------------------
text_vectorizer = TfidfVectorizer(
    max_features=1200,
    ngram_range=(1, 3),  # Include trigrams for better pattern matching
    stop_words='english',
    min_df=1,
    max_df=0.85,
    sublinear_tf=True,  # Use sublinear TF scaling
    smooth_idf=True     # Smooth IDF weights
)

# Enhanced pipeline
pipeline = Pipeline([
    ('tfidf', text_vectorizer),
    ('nn', NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute'))
])

print("🔄 Building ENHANCED similarity search engine...")
X = df['Search_Text_Enhanced']
pipeline.fit(X)
print("✅ ENHANCED similarity engine ready!")

# ---------------------------
# 5️⃣ SMART PREDICTION WITH CONTEXT MATCHING
# ---------------------------
def smart_predict_disease(crop, symptoms, region="All India"):
    """
    Enhanced prediction with multiple strategies
    """
    # Strategy 1: Enhanced similarity search
    symptoms_clean = enhanced_clean_symptoms(symptoms)
    input_text = f"{crop} {symptoms_clean} {region}"
    
    # Add context terms based on input
    if 'yellow' in symptoms_clean:
        input_text += ' yellowing chlorosis '
    if 'spot' in symptoms_clean:
        input_text += ' spotting lesion '
    if 'rot' in symptoms_clean:
        input_text += ' rotting decay '
    if 'wilt' in symptoms_clean:
        input_text += ' wilting drooping '
    
    try:
        # Transform input
        X_input = pipeline.named_steps['tfidf'].transform([input_text])
        
        if X_input.sum() == 0:  # No features matched
            return keyword_fallback(crop, symptoms_clean, region)
        
        # Find matches
        distances, indices = pipeline.named_steps['nn'].kneighbors(X_input)
        
        top_matches = []
        for i, idx in enumerate(indices[0]):
            match = df.iloc[idx]
            similarity = 1 - distances[0][i]
            
            # Boost similarity if crop matches exactly
            crop_boost = 0.2 if crop.lower() == match['Crop'].lower() else 0
            boosted_similarity = min(1.0, similarity + crop_boost)
            
            top_matches.append({
                'disease': match['Disease'],
                'similarity': round(boosted_similarity, 3),
                'crop': match['Crop'],
                'symptoms': match['Symptoms'],
                'organic_remedy': match['Organic_Remedy'],
                'chemical_remedy': match['Chemical_Remedy'],
                'prevention': match['Prevention'],
                'region': match['Region'],
                'season': match['Season']
            })
        
        # Apply confidence thresholds
        best_match = top_matches[0]
        
        if best_match['similarity'] > 0.4:
            confidence = 'high' if best_match['similarity'] > 0.6 else 'medium'
            return {
                'status': 'success',
                'best_match': best_match,
                'alternative_matches': top_matches[1:3],
                'confidence': confidence
            }
        else:
            return {
                'status': 'suggestions',
                'message': 'Top matches found:',
                'matches': top_matches[:3],
                'advice': 'Consider these possibilities and consult expert if unsure'
            }
            
    except Exception as e:
        return keyword_fallback(crop, symptoms_clean, region)

def keyword_fallback(crop, symptoms, region):
    """Improved keyword matching with symptom pattern recognition"""
    crop_matches = df[df['Crop'].str.lower() == crop.lower()]
    
    if len(crop_matches) == 0:
        return {
            'status': 'crop_not_found',
            'available_crops': df['Crop'].unique().tolist()
        }
    
    # Symptom pattern scoring
    scored_matches = []
    symptom_words = [word for word in symptoms.split() if len(word) > 3]
    
    for _, row in crop_matches.iterrows():
        score = 0
        disease_symptoms = row['Symptoms_enhanced']
        
        # Exact word matches
        for word in symptom_words:
            if word in disease_symptoms:
                score += 2  # Higher weight for exact matches
        
        # Partial matches and synonyms
        for word in symptom_words:
            if any(partial in disease_symptoms for partial in [word[:4], word[-4:]]):
                score += 0.5
        
        # Normalize score
        if symptom_words:
            score = score / (len(symptom_words) * 2)  # Max score of 1.0
        
        scored_matches.append((score, row))
    
    # Get top matches
    scored_matches.sort(reverse=True, key=lambda x: x[0])
    top_matches = [match for score, match in scored_matches if score > 0.1][:3]
    
    if top_matches:
        results = []
        for match in top_matches:
            results.append({
                'disease': match['Disease'],
                'match_score': round(score, 3),
                'symptoms': match['Symptoms'],
                'organic_remedy': match['Organic_Remedy'],
                'chemical_remedy': match['Chemical_Remedy'],
                'prevention': match['Prevention']
            })
        
        return {
            'status': 'keyword_match',
            'best_match': results[0],
            'other_options': results[1:] if len(results) > 1 else []
        }
    else:
        return {
            'status': 'no_confident_match',
            'message': 'No specific match found. Common diseases for this crop:',
            'common_diseases': crop_matches['Disease'].tolist()[:5]
        }

# ---------------------------
# 6️⃣ TEST THE ENHANCED SYSTEM
# ---------------------------
print("\n" + "="*70)
print("🚀 TESTING ENHANCED CROP DOCTOR AI SYSTEM")
print("="*70)

test_cases = [
    {
        "crop": "Rice",
        "symptoms": "yellow leaves with brown spots and diamond shaped lesions",
        "region": "Tamil Nadu",
        "expected": "Blast"  # What we expect
    },
    {
        "crop": "Tomato", 
        "symptoms": "dark brown spots with concentric rings on older leaves",
        "region": "Maharashtra",
        "expected": "Early Blight"
    },
    {
        "crop": "Mango",
        "symptoms": "black spots on fruits and white powdery growth on leaves",
        "region": "Uttar Pradesh", 
        "expected": "Anthracnose or Powdery Mildew"
    },
    {
        "crop": "Wheat",
        "symptoms": "orange brown pustules on leaves in stripes",
        "region": "Punjab",
        "expected": "Rust"
    },
    {
        "crop": "Cotton",
        "symptoms": "leaf curling and vein thickening",
        "region": "Gujarat",
        "expected": "Leaf Curl Virus"
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n🔍 Test Case {i}:")
    print(f"🌱 Crop: {test['crop']}")
    print(f"📝 Symptoms: {test['symptoms']}")
    print(f"🎯 Expected: {test['expected']}")
    print("-" * 50)
    
    result = smart_predict_disease(test['crop'], test['symptoms'], test['region'])
    
    if result['status'] == 'success':
        match = result['best_match']
        print(f"✅ DIAGNOSIS: {match['disease']} (Confidence: {result['confidence']})")
        print(f"📊 Similarity: {match['similarity']}")
        print(f"💡 Quick Fix: {match['organic_remedy'][:80]}...")
        
    elif result['status'] == 'suggestions':
        print("💡 Top suggestions:")
        for j, match in enumerate(result['matches'], 1):
            print(f"   {j}. {match['disease']} (similarity: {match['similarity']})")
            
    elif result['status'] == 'keyword_match':
        match = result['best_match']
        print(f"🔍 KEYWORD MATCH: {match['disease']}")
        print(f"💊 Remedy: {match['organic_remedy'][:80]}...")
    
    print("="*70)

# ---------------------------
# 7️⃣ QUICK PERFORMANCE STATS
# ---------------------------
print(f"\n📈 ENHANCED SYSTEM SUMMARY:")
print(f"• Diseases: {len(df)} | Crops: {df['Crop'].nunique()} | Regions: {df['Region'].nunique()}")
print(f"• Using enhanced text processing with synonym expansion")
print(f"• Similarity thresholds: >0.4 = suggestions, >0.6 = high confidence")

# ---------------------------
# 8️⃣ SAVE ENHANCED SYSTEM
# ---------------------------
joblib.dump(pipeline, 'crop_doctor_enhanced.pkl')
joblib.dump(df, 'crop_database_enhanced.pkl')

print("💾 Enhanced system saved!")

# ---------------------------
# 9️⃣ DEMO FUNCTION FOR HACKATHON
# ---------------------------
def demo_crop_doctor():
    """Perfect for your hackathon demo!"""
    print("\n🎯 CROP DOCTOR AI - LIVE DEMO")
    print("Enter crop symptoms and get instant diagnosis!\n")
    
    demo_inputs = [
        ("Rice", "yellow spots on leaves with diamond shape"),
        ("Tomato", "brown concentric circles on leaves"), 
        ("Mango", "white powder on leaves and fruits"),
        ("Wheat", "orange stripes on leaves")
    ]
    
    for crop, symptoms in demo_inputs:
        print(f"\n>>> Farmer Query: My {crop} has {symptoms}")
        result = smart_predict_disease(crop, symptoms)
        
        if result['status'] in ['success', 'keyword_match']:
            match = result['best_match']
            print(f"🤖 AI Diagnosis: {match['disease']}")
            print(f"💡 Solution: {match['organic_remedy']}")
        else:
            print("🤖 AI: Please consult local Krishi Vigyan Kendra for accurate diagnosis")
        print("-" * 60)

# Run the demo
# demo_crop_doctor()