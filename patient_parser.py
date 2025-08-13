import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional

# Define the data structure for the extracted patient information
class PatientData(BaseModel):
    age: int = Field(..., description="Patient's age in years.")
    sex: str = Field(..., description="Patient's sex (Male or Female).")
    race: str = Field(..., description="Patient's race (White, African American, or Other).")
    total_cholesterol: int = Field(..., description="Total cholesterol level.")
    hdl_cholesterol: int = Field(..., description="HDL cholesterol level.")
    systolic_bp: int = Field(..., description="Systolic blood pressure.")
    on_hypertension_treatment: bool = Field(..., description="Is the patient on hypertension treatment? (true/false)")
    smoker: bool = Field(..., description="Is the patient a smoker? (true/false)")
    diabetic: bool = Field(..., description="Is the patient diabetic? (true/false)")
    family_history: Optional[str] = Field("Unknown", description="Family history of premature ASCVD (Present, Absent, or Unknown).")
    elevated_ldl: Optional[str] = Field("Unknown", description="Persistently elevated LDL-C ≥160 mg/dL (Present, Absent, or Unknown).")
    ckd: Optional[str] = Field("Unknown", description="Chronic kidney disease (Present, Absent, or Unknown).")
    metabolic_syndrome: Optional[str] = Field("Unknown", description="Metabolic syndrome (Present, Absent, or Unknown).")
    inflammation: Optional[str] = Field("Unknown", description="Inflammation (elevated hsCRP ≥2.0 mg/L) (Present, Absent, or Unknown).")
    elevated_triglycerides: Optional[str] = Field("Unknown", description="Persistently elevated triglycerides (≥175 mg/dL) (Present, Absent, or Unknown).")
    premature_menopause: Optional[str] = Field("Unknown", description="Premature menopause (Present, Absent, or Unknown).")
    ethnicity_risk: Optional[str] = Field("Unknown", description="Ethnicity-specific risk factors (Present, Absent, or Unknown).")
    medications: Optional[list[str]] = Field([], description="A list of the patient's current medications.")


def parse_patient_description(description: str) -> dict:
    """
    Parses a free-text patient description to extract structured clinical data.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    structured_llm = llm.with_structured_output(PatientData)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert clinical data extraction assistant. Your task is to extract key clinical parameters from a patient's description. The user will provide a text description of a patient. You must extract the information and format it according to the provided schema. If a value is not mentioned, use a default or 'Unknown' where appropriate."),
            ("human", "{description}")
        ]
    )

    chain = prompt | structured_llm
    
    try:
        extracted_data = chain.invoke({"description": description})
        return extracted_data.dict()
    except Exception as e:
        print(f"Error during data extraction: {e}")
        return None
