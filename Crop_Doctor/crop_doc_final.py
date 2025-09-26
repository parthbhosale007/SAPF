# crop_doctor_simple.py - Simple version for your team
import joblib
import pandas as pd
import re

# Load your enhanced model and database
model = joblib.load('Crop_Doctor/crop_doctor_enhanced.pkl')
database = pd.read_csv('Crop_Doctor/crop_diseases.csv')

def predict_crop_issue(crop_name, issue_text, location="All India"):
    """
    SIMPLE 3-INPUT FUNCTION FOR YOUR TEAM
    Input: crop name, issue description, location
    Output: Diagnosis and treatment
    """
    # Clean the inputs
    crop_name = crop_name.strip().title()
    issue_text = issue_text.lower().strip()
    location = location.strip()
    
    # Create search query (similar to your enhanced function)
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s.,;-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    search_query = f"{crop_name} {clean_text(issue_text)} {location}"
    
    try:
        # Use the enhanced model to find matches
        X_input = model.named_steps['tfidf'].transform([search_query])
        distances, indices = model.named_steps['nn'].kneighbors(X_input)
        
        # Get the best match
        best_match_idx = indices[0][0]
        best_match = database.iloc[best_match_idx]
        similarity_score = 1 - distances[0][0]
        
        # Simple output format for your team
        return {
            'success': True,
            'crop': crop_name,
            'issue': issue_text,
            'location': location,
            'diagnosed_disease': best_match['Disease'],
            'confidence_score': round(similarity_score, 3),
            'organic_treatment': best_match['Organic_Remedy'],
            'chemical_treatment': best_match['Chemical_Remedy'],
            'prevention_methods': best_match['Prevention'],
            'common_symptoms': best_match['Symptoms'],
            'season_advice': best_match['Season'],
            'region_advice': best_match['Region']
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Please try with different symptoms or consult agriculture expert'
        }

# Bonus function: Get list of available crops
def get_supported_crops():
    return sorted(database['Crop'].unique().tolist())

def test(crop_name="Tomato", issue_text="brown spots with concentric rings on leaves", location="Maharashtra"):
    
    result = predict_crop_issue(
        crop_name=crop_name,
        issue_text=issue_text, 
        location=location
    )
    
    if result['success']:
        print("✅ TEST SUCCESSFUL!")
        print(f"Diagnosis: {result['diagnosed_disease']}")
        print(f"Confidence: {result['confidence_score']}")
        print(f"Organic Treatment: {result['organic_treatment']}")
    else:
        print("❌ Test failed:", result['message'])

# Test the function
if __name__ == "__main__":
    # result = predict_crop_issue(crop_name="Tomato", issue_text="brown spots with concentric rings on leaves", location="Maharashtra")
    pass
    # test()
    