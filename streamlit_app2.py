print("Starting Streamlit app...")
import streamlit as st
import json
print("Imported streamlit and json")

from patient_db import initialize_db, Patient, Medication
print("Imported patient_db")
from ascvd_pce import ascvd_pce
print("Imported ascvd_pce")
from rag_pipeline import initialize_rag_pipeline, run_gap_analysis, get_bhc_recommendations
print("Imported rag_pipeline")
from patient_parser import parse_patient_description
print("Imported patient_parser")

# Initialize the RAG pipeline once and cache it
try:
    rag_agent = initialize_rag_pipeline()
    print("RAG pipeline initialized.")
except ValueError as e:
    st.error(f"Failed to initialize the RAG pipeline: {e}")
    st.stop()


st.set_page_config(page_title="ASCVD Risk Assistant", layout="centered")
print("Page config set.")

st.title("🩺 ASCVD Clinical AI Assistant")
print("Title set.")

# Ensure table exists
initialize_db()
print("Database initialized.")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["➕ Add Patient", "📋 View All Patients", "Clinical Analysis"])
print(f"Menu option selected: {menu}")

# Add Patient Form
if menu == "➕ Add Patient":
    st.subheader("Enter New Patient Information")

    with st.form("add_patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            first_name = st.text_input("First Name")
            age = st.number_input("Age", min_value=40, max_value=79)
            sex = st.selectbox("Sex", ["Male", "Female"])
            total_cholesterol = st.number_input("Total Cholesterol")
            sbp = st.number_input("Systolic Blood Pressure")
            smoker = st.selectbox("Smoker?", ["Yes", "No"])

        with col2:
            last_name = st.text_input("Last Name")
            race = st.selectbox("Race", ["White", "African American", "Other"])
            hdl = st.number_input("HDL Cholesterol")
            bp_treatment = st.selectbox("On Hypertension Treatment?", ["Yes", "No"])
            diabetic = st.selectbox("Diabetic?", ["Yes", "No"])

        st.subheader("Risk Enhancers")
        col3, col4 = st.columns(2)
        with col3:
            family_history = st.selectbox("Family history of premature ASCVD?", ["Present", "Absent", "Unknown"])
            elevated_ldl = st.selectbox("Persistently elevated LDL-C ≥160 mg/dL?", ["Present", "Absent", "Unknown"])
            ckd = st.selectbox("Chronic kidney disease?", ["Present", "Absent", "Unknown"])
            metabolic_syndrome = st.selectbox("Metabolic syndrome?", ["Present", "Absent", "Unknown"])
        with col4:
            inflammation = st.selectbox("Inflammation (elevated hsCRP ≥2.0 mg/L)?", ["Present", "Absent", "Unknown"])
            elevated_triglycerides = st.selectbox("Persistently elevated triglycerides (≥175 mg/dL)?", ["Present", "Absent", "Unknown"])
            premature_menopause = st.selectbox("Premature menopause?", ["Present", "Absent", "Unknown"])
            ethnicity_risk = st.selectbox("Ethnicity-specific risk factors?", ["Present", "Absent", "Unknown"])

        submitted = st.form_submit_button("Add Patient")
        if submitted:
            # Convert checkbox values from Yes/No to 1/0
            bp_treatment_val = 1 if bp_treatment == "Yes" else 0
            smoker_val = 1 if smoker == "Yes" else 0
            diabetic_val = 1 if diabetic == "Yes" else 0

            new_patient = Patient(
                first_name=first_name,
                last_name=last_name,
                age=age,
                sex=sex,
                race=race,
                total_cholesterol=total_cholesterol,
                hdl_cholesterol=hdl,
                systolic_bp=sbp,
                on_hypertension_treatment=bp_treatment_val,
                smoker=smoker_val,
                diabetic=diabetic_val,
                family_history=family_history,
                elevated_ldl=elevated_ldl,
                ckd=ckd,
                metabolic_syndrome=metabolic_syndrome,
                inflammation=inflammation,
                elevated_triglycerides=elevated_triglycerides,
                premature_menopause=premature_menopause,
                ethnicity_risk=ethnicity_risk
            )
            new_patient.save()
            st.success(f"✅ {first_name} {last_name} added successfully!")

# View Patients Table
elif menu == "📋 View All Patients":
    st.subheader("📁 Patient Records")
    patients = Patient.get_all()

    if not patients:
        st.info("No patient records found.")
    else:
        # Create a list of dictionaries for st.dataframe
        patient_data = [
            {
                "ID": p.id, "First": p.first_name, "Last": p.last_name, "Age": p.age, "Sex": p.sex,
                "Race": p.race, "Total Cholesterol": p.total_cholesterol, "HDL": p.hdl_cholesterol,
                "BP": p.systolic_bp, "BP Rx": "Yes" if p.on_hypertension_treatment else "No",
                "Smoker": "Yes" if p.smoker else "No", "Diabetic": "Yes" if p.diabetic else "No",
                "Family History": p.family_history, "Elevated LDL": p.elevated_ldl, "CKD": p.ckd,
                "Metabolic Syndrome": p.metabolic_syndrome, "Inflammation": p.inflammation,
                "Elevated Triglycerides": p.elevated_triglycerides,
                "Premature Menopause": p.premature_menopause, "Ethnicity Risk": p.ethnicity_risk
            }
            for p in patients
        ]
        st.dataframe(patient_data, use_container_width=True)

        st.subheader("💊 Current Medications")
        patient_dict = {f"{p.id} - {p.first_name} {p.last_name}": p for p in patients}
        selected_label = st.selectbox("Select a patient to view medications:", list(patient_dict.keys()))
        selected_patient = patient_dict[selected_label]

        meds = Medication.get_by_patient(selected_patient.id)

        if meds:
            for med in meds:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"{med.drug_name} {med.dose} {med.frequency}")
                with col2:
                    if st.button(f"Delete {med.med_id}", key=f"delete_{med.med_id}"):
                        med.delete()
                        st.rerun()
        else:
            st.info("No medications recorded for this patient.")

        with st.form("add_med_form"):
            st.subheader("Add New Medication")
            drug_name = st.text_input("Drug Name")
            dose = st.text_input("Dose")
            frequency = st.text_input("Frequency")
            submitted = st.form_submit_button("Add Medication")
            if submitted:
                new_med = Medication(
                    patient_id=selected_patient.id,
                    drug_name=drug_name,
                    dose=dose,
                    frequency=frequency
                )
                new_med.save()
                st.success("Medication added.")
                st.rerun()


elif menu == "Clinical Analysis":
    st.subheader("Clinical Analysis")

    input_method = st.radio("How would you like to provide patient data?",
                            ("Select Patient from Database", "Describe Patient in Text"))

    if input_method == "Select Patient from Database":
        patients = Patient.get_all()

        if not patients:
            st.info("No patients in database. Please add one first.")
        else:
            patient_dict = {f"{p.id} - {p.first_name} {p.last_name} (Age {p.age})": p for p in patients}
            selected_label = st.selectbox("Select a patient to assess:", list(patient_dict.keys()))
            patient = patient_dict[selected_label]

            st.markdown("### 🧑 Patient Details")
            st.write(f"**ID:** {patient.id}")
            st.write(f"**Name:** {patient.first_name} {patient.last_name}")
            st.write(f"**Age:** {patient.age}")
            st.write(f"**Sex:** {patient.sex}")
            st.write(f"**Race:** {patient.race}")
            st.write(f"**Total Cholesterol:** {patient.total_cholesterol}")
            st.write(f"**HDL Cholesterol:** {patient.hdl_cholesterol}")
            st.write(f"**Systolic BP:** {patient.systolic_bp}")
            st.write(f"**On Hypertension Treatment:** {'Yes' if patient.on_hypertension_treatment else 'No'}")
            st.write(f"**Smoker:** {'Yes' if patient.smoker else 'No'}")
            st.write(f"**Diabetic:** {'Yes' if patient.diabetic else 'No'}")

            st.markdown("### 💊 Current Medications")
            meds = Medication.get_by_patient(patient.id)
            if meds:
                for med in meds:
                    st.write(f"- {med.drug_name} {med.dose} {med.frequency}")
            else:
                st.write("No medications recorded.")

            st.markdown("### Risk Enhancers")
            st.write(f"**Family history of premature ASCVD:** {patient.family_history}")
            st.write(f"**Persistently elevated LDL-C ≥160 mg/dL:** {patient.elevated_ldl}")
            st.write(f"**Chronic kidney disease:** {patient.ckd}")
            st.write(f"**Metabolic syndrome:** {patient.metabolic_syndrome}")
            st.write(f"**Inflammation (elevated hsCRP ≥2.0 mg/L):** {patient.inflammation}")
            st.write(f"**Persistently elevated triglycerides (≥175 mg/dL):** {patient.elevated_triglycerides}")
            st.write(f"**Premature menopause:** {patient.premature_menopause}")
            st.write(f"**Ethnicity-specific risk factors:** {patient.ethnicity_risk}")

            st.markdown("---")

            if st.button("Analyze"):
                with st.spinner("Analyzing patient data and generating recommendations..."):
                    try:
                        risk_percent = ascvd_pce(
                            age=patient.age,
                            sex=patient.sex,
                            race=patient.race,
                            tc=patient.total_cholesterol,
                            hdl=patient.hdl_cholesterol,
                            sbp=patient.systolic_bp,
                            treated=patient.on_hypertension_treatment,
                            smoker=patient.smoker,
                            diabetic=patient.diabetic
                        )

                        st.success(f"🧠 Estimated 10-year ASCVD Risk: **{risk_percent:.2f}%**")

                        if risk_percent < 5:
                            risk_category = "Low Risk (< 5%)"
                        elif 5 <= risk_percent < 7.5:
                            risk_category = "Borderline Risk (5% to 7.4%)"
                        elif 7.5 <= risk_percent < 20:
                            risk_category = "Intermediate Risk (7.5% to 19.9%)"
                        else:
                            risk_category = "High Risk (≥ 20%)"
                        
                        meds = Medication.get_by_patient(patient.id)
                        med_list = [f"{med.drug_name} {med.dose} {med.frequency}" for med in meds]
                        med_str = "\n".join(med_list) if med_list else "None"

                        prompt = (
                            f"Generate a detailed clinical analysis for the following patient. "
                            f"The patient's estimated 10-year ASCVD risk is **{risk_percent:.2f}%**, placing them in the **{risk_category}** category.\n\n"
                            f"**Patient Data:**\n{json.dumps(patient.__dict__, default=str, indent=2)}\n\n"
                            f"**Current Medications:**\n{med_str}\n\n"
                            f"Based on this information, please perform a full clinical analysis and provide a treatment and follow-up plan as per the instructions."
                        )

                    except (TypeError, ValueError) as e:
                        st.warning(f"Could not calculate exact ASCVD risk due to missing data: {e}. Providing a qualitative assessment instead.")
                        prompt = (
                            f"A precise 10-year ASCVD risk score could not be calculated for the following patient due to missing information. "
                            f"Based on the available data, please provide a qualitative risk assessment (e.g., 'likely low risk', 'potentially high risk').\n\n"
                            f"**Patient Data:**\n{json.dumps(patient.__dict__, default=str, indent=2)}\n\n"
                            f"Generate a detailed clinical analysis, including specific medication and lifestyle recommendations, based on this qualitative risk assessment and the available patient data."
                        )
                    
                    # Display BHC match info
                    bhc_recs = get_bhc_recommendations(patient.id)
                    if "error" not in bhc_recs:
                        st.markdown("### Matched BHC Patient Profile")
                        st.write(f"**DB Patient Info:** {bhc_recs['db_patient_info']}")
                        st.write(f"**Matched BHC Patient Info:** {bhc_recs['matched_bhc_patient_info']}")
                    
                    # Run full analysis
                    result = run_gap_analysis(prompt, patient.id)
                    report = result["report"]
                    sources = result["sources"]

                st.markdown("### Clinical Analysis Report")
                st.write(report)

                if sources:
                    st.markdown("### Sources")
                    for source in sources:
                        st.write(source)

    elif input_method == "Describe Patient in Text":
        patient_description = st.text_area("Describe the patient's clinical details:", height=200,
                                           placeholder="e.g., 55-year-old male, non-smoker, not diabetic, with a total cholesterol of 220, HDL of 45, and systolic BP of 130. He is not on any hypertension medication. He has a family history of heart disease.")
        if st.button("Analyze from Text"):
            if patient_description:
                with st.spinner("Parsing patient description..."):
                    patient_data = parse_patient_description(patient_description)

                if patient_data:
                    st.success("Patient data extracted successfully.")
                    st.markdown("### 🧑 Extracted Patient Details")
                    # Display extracted details...
                    st.write(f"**Age:** {patient_data.get('age', 'N/A')}")
                    st.write(f"**Sex:** {patient_data.get('sex', 'N/A')}")
                    st.write(f"**Race:** {patient_data.get('race', 'N/A')}")
                    st.write(f"**Total Cholesterol:** {patient_data.get('total_cholesterol', 'N/A')}")
                    st.write(f"**HDL Cholesterol:** {patient_data.get('hdl_cholesterol', 'N/A')}")
                    st.write(f"**Systolic BP:** {patient_data.get('systolic_bp', 'N/A')}")
                    st.write(f"**On Hypertension Treatment:** {'Yes' if patient_data.get('on_hypertension_treatment') else 'No'}")
                    st.write(f"**Smoker:** {'Yes' if patient_data.get('smoker') else 'No'}")
                    st.write(f"**Diabetic:** {'Yes' if patient_data.get('diabetic') else 'No'}")
                    st.markdown("---")

                    try:
                        risk_percent = ascvd_pce(
                            age=patient_data['age'],
                            sex=patient_data['sex'],
                            race=patient_data['race'],
                            tc=patient_data['total_cholesterol'],
                            hdl=patient_data['hdl_cholesterol'],
                            sbp=patient_data['systolic_bp'],
                            treated=patient_data['on_hypertension_treatment'],
                            smoker=patient_data['smoker'],
                            diabetic=patient_data['diabetic']
                        )

                        st.success(f"🧠 Estimated 10-year ASCVD Risk: **{risk_percent:.2f}%**")

                        if risk_percent < 5:
                            risk_category = "Low Risk (< 5%)"
                        elif 5 <= risk_percent < 7.5:
                            risk_category = "Borderline Risk (5% to 7.4%)"
                        elif 7.5 <= risk_percent < 20:
                            risk_category = "Intermediate Risk (7.5% to 19.9%)"
                        else:
                            risk_category = "High Risk (≥ 20%)"

                        prompt = (
                            f"Generate a detailed clinical analysis for the following patient. "
                            f"The patient's estimated 10-year ASCVD risk is **{risk_percent:.2f}%**, placing them in the **{risk_category}** category.\n\n"
                            f"**Patient Data:**\n{json.dumps(patient_data, indent=2)}\n\n"
                            f"Based on this information, please perform a full clinical analysis and provide a treatment and follow-up plan as per the instructions."
                        )

                    except (TypeError, ValueError, KeyError) as e:
                        st.warning(f"Could not calculate exact ASCVD risk due to missing data. Providing a qualitative assessment instead.")
                        prompt = (
                            f"A precise 10-year ASCVD risk score could not be calculated for the following patient due to missing information. "
                            f"Based on the available data, please provide a qualitative risk assessment (e.g., 'likely low risk', 'potentially high risk').\n\n"
                            f"**Patient Data:**\n{json.dumps(patient_data, indent=2)}\n\n"
                            f"Generate a detailed clinical analysis, including specific medication and lifestyle recommendations, based on this qualitative risk assessment and the available patient data."
                        )

                    with st.spinner("Generating clinical analysis report..."):
                        result = run_gap_analysis(prompt)
                        report = result["report"]
                        sources = result["sources"]

                    st.markdown("### Clinical Analysis Report")
                    st.write(report)

                    if sources:
                        st.markdown("### Sources")
                        for source in sources:
                            st.write(source)
                else:
                    st.error("Could not parse the patient description. Please provide more details or check the format.")
            else:
                st.warning("Please describe the patient.")

print("App finished.")
