

import pandas as pd
import patient_db
import re

def get_patient_data_from_csv(file_path):
    """
    Reads the patient data from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def get_patient_data_from_db():
    """
    Reads the patient data from the database.
    """
    patients = patient_db.Patient.get_all()
    return patients

def parse_csv_input(input_text):
    """
    Parses the input text from the CSV to extract patient information.
    """
    patient_info = {}
    
    # Extract sex
    sex_match = re.search(r'<SEX>\s*(\w)', input_text)
    if sex_match:
        sex_full = 'Male' if sex_match.group(1).upper() == 'M' else 'Female'
        patient_info['sex'] = sex_full
    
    # More robust age extraction
    age_match = re.search(r'(\d+)\s*y\.o\.', input_text) or \
                re.search(r'(\d+)\s*yo', input_text) or \
                re.search(r'(\d+)\s*year old', input_text)
    if age_match:
        patient_info['age'] = int(age_match.group(1))
    else:
        patient_info['age'] = 'Unknown'

    # Extract risk factors
    patient_info['smoker'] = 1 if re.search(r'smoker|tobacco use', input_text, re.IGNORECASE) else 0
    patient_info['diabetic'] = 1 if re.search(r'diabetic|DM', input_text, re.IGNORECASE) else 0
    patient_info['on_hypertension_treatment'] = 1 if re.search(r'hypertension|HTN', input_text, re.IGNORECASE) else 0
        
    return patient_info

def find_best_match(db_patient, csv_patients_data):
    """
    Finds the best match for a database patient from the CSV data.
    """
    best_match = None
    max_score = -1

    for index, csv_row in csv_patients_data.iterrows():
        csv_patient_info = parse_csv_input(csv_row['input'])
        
        score = 0
        if 'sex' in csv_patient_info and csv_patient_info['sex'] == db_patient.sex:
            score += 1
        if csv_patient_info.get('smoker') == db_patient.smoker:
            score += 1
        if csv_patient_info.get('diabetic') == db_patient.diabetic:
            score += 1
        if csv_patient_info.get('on_hypertension_treatment') == db_patient.on_hypertension_treatment:
            score += 1
        if csv_patient_info.get('age') != 'Unknown' and abs(csv_patient_info.get('age', 0) - db_patient.age) <= 5:
            score += 2 # Higher weight for age match
            
        if score > max_score:
            max_score = score
            best_match = (csv_row, csv_patient_info)
            
    return best_match

def summarize_recommendations(target_text):
    """
    Summarizes the target text into a concise list of recommendations.
    """
    recommendations = []
    
    # Medications
    med_keywords = ['aspirin', 'plavix', 'atorvastatin', 'lisinopril', 'metoprolol', 'ticagrelor', 'rivaroxaban']
    for med in med_keywords:
        if re.search(r'\b' + med + r'\b', target_text, re.IGNORECASE):
            recommendations.append(f"Consider {med.title()} for treatment.")

    # Lifestyle
    if re.search(r'smoking cessation', target_text, re.IGNORECASE):
        recommendations.append("Recommend smoking cessation counseling.")
    if re.search(r'cardiac rehabilitation', target_text, re.IGNORECASE):
        recommendations.append("Recommend cardiac rehabilitation.")

    # Follow-up
    if re.search(r'follow up with cardiology', target_text, re.IGNORECASE):
        recommendations.append("Schedule follow-up with cardiology.")
    if re.search(r'check TSH', target_text, re.IGNORECASE):
        recommendations.append("Schedule follow-up to check TSH.")

    if not recommendations:
        return "No specific recommendations could be automatically extracted. Please review the full text for details."

    return "\n".join(f"- {rec}" for rec in recommendations)


def compare_data(csv_data, db_patients):
    """
    Compares the data from the CSV file and the database and provides justification.
    """
    print("Generating Analysis and Recommendations...\n")
    
    for db_patient in db_patients:
        best_match_row, best_match_info = find_best_match(db_patient, csv_data)
        
        print(f"--- Patient: {db_patient.first_name} {db_patient.last_name} ---")
        print(f"DB Patient Info: Age {db_patient.age}, Sex {db_patient.sex}, Smoker: {'Yes' if db_patient.smoker else 'No'}, Diabetic: {'Yes' if db_patient.diabetic else 'No'}, Hypertension: {'Yes' if db_patient.on_hypertension_treatment else 'No'}")
        
        matched_age = best_match_info.get('age', 'N/A')
        print(f"\nMatched BHC Patient Info: Age {matched_age}, Sex {best_match_info.get('sex', 'N/A')}, Smoker: {'Yes' if best_match_info.get('smoker') else 'No'}, Diabetic: {'Yes' if best_match_info.get('diabetic') else 'No'}, Hypertension: {'Yes' if best_match_info.get('on_hypertension_treatment') else 'No'}")
        
        print(f"\nJustification from BHC Text (Full Outcome):")
        print(best_match_row['target'])
        
        print(f"\nSummarized Recommendations:")
        print(summarize_recommendations(best_match_row['target']))
        print("-" * 40 + "\n")


if __name__ == "__main__":
    csv_file_path = 'data_guidelines/ascvd_ranked_200.csv'
    
    # Get data from CSV
    csv_data = get_patient_data_from_csv(csv_file_path)
    
    # Get data from DB
    db_data = get_patient_data_from_db()
    
    # Compare data and provide justification
    compare_data(csv_data, db_data)

