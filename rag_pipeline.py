from dotenv import load_dotenv
import os
import re
import pandas as pd
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
)
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import json
import shutil
import patient_db

load_dotenv()

def get_loader_type(file_path):
    """Determine the appropriate loader based on file extension."""
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif file_path.endswith(".md"):
        return TextLoader(file_path, encoding="latin-1")
    elif file_path.endswith(".html") or file_path.endswith(".htm"):
        return UnstructuredHTMLLoader(file_path)
    elif file_path.endswith(".csv"):
        return CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def load_documents_from_folder(folder_path, source_map_path):
    """Loads documents from a folder, including the BHC text."""
    all_documents = []
    print(f"Starting to load documents from: {folder_path}")
    with open(source_map_path, 'r') as f:
        source_map = json.load(f)

    file_list = os.listdir(folder_path)
    print(f"Files found in directory: {file_list}")

    for fname in file_list:
        full_path = os.path.join(folder_path, fname)
        if not os.path.isfile(full_path):
            print(f"Skipping non-file item: {fname}")
            continue

        print(f"Processing file: {fname}")
        try:
            loader = get_loader_type(full_path)
            docs = loader.load()
            if not docs:
                print(f"Warning: No documents were loaded from {fname}.")
                continue

            # Get the source URL from the map
            source_url = source_map.get(fname)

            # Add source and file name to each document's metadata
            if source_url:
                for doc in docs:
                    doc.metadata["source"] = source_url
                    doc.metadata["file_name"] = fname
                print(f"Successfully loaded {len(docs)} documents from {fname} with source: {source_url}")
            else:
                for doc in docs:
                    doc.metadata["source"] = "Unknown source"
                    doc.metadata["file_name"] = fname
                print(f"Warning: No source URL found for {fname} in the source map.")

            all_documents.extend(docs)
        except Exception as e:
            print(f"ERROR processing file {fname}: {e}")
            
    # Specifically load the BHC text
    bhc_file_path = 'data_guidelines/ascvd_ranked_200.csv'
    if os.path.exists(bhc_file_path):
        print("Loading BHC text...")
        try:
            loader = CSVLoader(bhc_file_path)
            bhc_docs = loader.load()
            for doc in bhc_docs:
                doc.metadata["source"] = "BHC Text"
            all_documents.extend(bhc_docs)
            print(f"Successfully loaded {len(bhc_docs)} documents from BHC text.")
        except Exception as e:
            print(f"ERROR processing BHC file: {e}")


    print(f"Finished loading. Total documents loaded: {len(all_documents)}")
    return all_documents

def get_patient_data_from_csv(file_path):
    """
    Reads the patient data from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

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
    age_match = re.search(r'(\d+)\s*y\\.o\\.', input_text) or \
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

def get_bhc_recommendations(patient_id: int):
    """
    Generates BHC-based recommendations for a given patient.
    """
    db_patient = patient_db.Patient.get(patient_id)
    if not db_patient:
        return {"error": "Patient not found."}

    csv_file_path = 'data_guidelines/ascvd_ranked_200.csv'
    csv_data = get_patient_data_from_csv(csv_file_path)
    
    best_match_row, best_match_info = find_best_match(db_patient, csv_data)
    
    recommendations = summarize_recommendations(best_match_row['target'])
    
    return {
        "db_patient_info": f"Age {db_patient.age}, Sex {db_patient.sex}, Smoker: {'Yes' if db_patient.smoker else 'No'}, Diabetic: {'Yes' if db_patient.diabetic else 'No'}, Hypertension: {'Yes' if db_patient.on_hypertension_treatment else 'No'}",
        "matched_bhc_patient_info": f"Age {best_match_info.get('age', 'N/A')}, Sex {best_match_info.get('sex', 'N/A')}, Smoker: {'Yes' if best_match_info.get('smoker') else 'No'}, Diabetic: {'Yes' if best_match_info.get('diabetic') else 'No'}, Hypertension: {'Yes' if best_match_info.get('on_hypertension_treatment') else 'No'}",
        "recommendations": recommendations
    }

def initialize_rag_pipeline():
    """Initializes the RAG pipeline, including the vector store and agent."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "chromadb"
    collection_name = "Cardiovascular_Guidelines"

    # Delete the old database directory if it exists
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print("Removed existing persist directory.")
        except Exception as e:
            print(f"ERROR deleting persist directory {persist_directory}: {e}")


    # Initialize the Chroma client
    os.makedirs(persist_directory, exist_ok = True)
    client = chromadb.Client()

    # Get the collection, creating it if it doesn't exist
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # Check if the database is empty
    if vectorstore._collection.count() == 0:
        folder_path = os.getenv("DATA_DIR")
        source_map_path = os.getenv("DATA_SOURCES")

        if not folder_path or not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not source_map_path or not os.path.exists(source_map_path):
            raise FileNotFoundError(f"Source map not found: {source_map_path}")

        documents = load_documents_from_folder(folder_path, source_map_path)

        if not documents:
            raise ValueError(f"No documents loaded from the folder: {folder_path}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        pages_split = text_splitter.split_documents(documents)

        # Add documents in batches to avoid exceeding ChromaDB's batch size limit
        batch_size = 100
        for i in range(0, len(pages_split), batch_size):
            batch = pages_split[i:i + batch_size]
            vectorstore.add_documents(batch)

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 15, 'fetch_k': 50})

    @tool
    def retriever_tool(query: str) -> str:
        """
        This tool searches and returns specific information from the clinical guidelines and BHC text,
        including details on statin intensity (high, moderate), recommended dosages, and follow-up timelines.
        Use this tool to find evidence-based recommendations for patient treatment plans, including real-world patient outcomes from the BHC text.
        """
        docs = retriever.invoke(query)
        if not docs:
            return json.dumps({"content": "I found no relevant information from the Clinical Guidelines provided for the query.", "sources": []})

        content_with_sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown source")
            content_with_sources.append(f"Source: {source}\nContent: {doc.page_content}")

        final_content = "\n\n---\n\n".join(content_with_sources)
        sources = list(set([doc.metadata.get("source", "Unknown source") for doc in docs]))

        return json.dumps({"content": final_content, "sources": sources})

    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        sources: list[str]

    def should_continue(state: AgentState):
        return hasattr(state['messages'][-1], 'tool_calls') and len(state['messages'][-1].tool_calls) > 0

    system_prompt = """
    You are a clinical decision support assistant. Your role is to analyze a patient's cardiovascular risk and provide treatment recommendations aligned with the latest clinical guidelines for ASCVD prevention and management and the BHC text.

    You will be given excerpts from clinical guidelines and the BHC text. Each excerpt is preceded by its source URL or "BHC Text".

    You must perform the following steps:
    1.  **Analyze Patient Data:** Review the patient's age, sex, race, cholesterol levels, blood pressure, and diabetic status to understand their baseline risk profile.
    2.  **Incorporate Risk Enhancers:** Consider all provided risk enhancers (e.g., family history, CKD, metabolic syndrome) as they are critical for refining treatment decisions, especially for patients in borderline or intermediate risk categories.
    3.  **Assess Current Medications:** Evaluate the patient's current medication list.
    4.  **Query Clinical Guidelines and BHC Text:** Use the `retriever_tool` to find the specific guideline recommendations for statin intensity (e.g., "high-intensity statin," "moderate-intensity statin") and follow-up plans based on the patient's calculated risk category and other clinical factors. Also, use the tool to find relevant patient cases and outcomes in the BHC text.
    5.  **Synthesize Recommendations:** Generate a comprehensive clinical analysis. When you use information from the retrieved excerpts, you MUST cite it by embedding the source URL or "BHC Text" from the 'Source:' line directly into the text using the format <CITATION:source_url_or_BHC_Text>. Do not number the citations yourself.

    Your response must include:
    - A clear statement on whether the patient's current treatment is aligned with the guidelines.
    - **Specific medication recommendations, including drug names and dosages (e.g., "initiate Atorvastatin 40mg daily" instead of "initiate moderate-intensity statin").**
    - **A detailed justification for your choices, linking them directly to the patient’s condition, risk score, and the evidence from the clinical guidelines and BHC text.** Explain *why* the recommended drug and dosage are appropriate, and reference similar cases from the BHC text where applicable.
    - An explanation of why your recommendation is preferable to alternative or current therapies.
    - A detailed follow-up plan, including timelines for reassessment and monitoring for efficacy and side effects.

    Always end your entire response with the disclaimer: "This is not a substitute for professional medical advice."
    """

    tools_dict = {tool.name: tool for tool in tools}

    def call_llm(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
        message = llm_with_tools.invoke(messages)
        return {'messages': [message]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        sources = []
        for t in tool_calls:
            if t['name'] not in tools_dict:
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                tool_output = tools_dict[t['name']].invoke(t['args'].get('query', ''))
                tool_data = json.loads(tool_output)
                result = tool_data["content"]
                sources.extend(tool_data["sources"])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results, 'sources': sources}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    
    return graph.compile()

def process_citations(report_with_placeholders: str) -> dict:
    import re
    
    cited_urls = sorted(list(set(re.findall(r"<CITATION:(.*?)>", report_with_placeholders))))
    
    if not cited_urls:
        return {"report": report_with_placeholders, "sources": []}
        
    citation_map = {url: i + 1 for i, url in enumerate(cited_urls)}
    
    final_report = report_with_placeholders
    for url, number in citation_map.items():
        final_report = final_report.replace(f"<CITATION:{url}>", f" [{number}]")
        
    formatted_sources = [f"[{number}] {url}" for url, number in citation_map.items()]
    
    return {"report": final_report, "sources": formatted_sources}

def run_gap_analysis(user_query: str, patient_id: int = None) -> dict:
    """
    Invoke the RAG agent on a single user_query and return the final answer and sources.
    """
    if patient_id:
        bhc_recs = get_bhc_recommendations(patient_id)
        if "error" not in bhc_recs:
            user_query += f"\n\n**BHC-Based Recommendations for a similar patient:**\n{bhc_recs['recommendations']}"

    rag_agent = initialize_rag_pipeline()
    state = {"messages": [HumanMessage(content=user_query)], "sources": []}
    out = rag_agent.invoke(state)
    
    raw_report = out["messages"][-1].content
    
    processed_result = process_citations(raw_report)
    
    return processed_result


