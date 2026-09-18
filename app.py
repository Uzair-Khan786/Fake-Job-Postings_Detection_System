import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib
import re

st.set_page_config(page_title="🎯 AI Job Analyzer", page_icon="⚙", layout="wide")

st.markdown(
    """
    <style>
    .fraud-tag {
        background-color: #15262c;    /* Dark slate background */
        color: #ffffff;               /* Crisp white text */
        border: 1px solid #2d4a53;    /* Distinct thin outer border line */
        padding: 5px 12px;
        border-radius: 20px;          /* Clean rounded pill shape */
        margin: 4px 3px; 
        font-size: 0.85em; 
        display: inline-block;
        font-family: sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    try :
        with open("models/vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        with open("models/SVM_model.pkl", "rb") as f:
            model = pickle.load(f)

        return vectorizer, model

    except FileNotFoundError :
        return None,None


vectorizer, model = load_model()

# Rule Constants
URGENCY_WORDS = ["urgent", "immediate", "asap", "act now", "limited time", "hurry"]
OTHER_TERMS = ["unlimited income","financial freedom","work when you want"]
HIGH_RISK_TITLES = ["data entry","home based","earn daily","administrative"]

def extract_indicators(data_dict) :

    indicators = []
    text_lower = str(data_dict.get('description', '')).lower()
    title_lower = str(data_dict.get('title', '')).lower()

    if any(w in text_lower for w in URGENCY_WORDS):
        indicators.append("Urgent Tone")

    if "no experience" in text_lower:
        indicators.append("No Experience Required")

    if len(text_lower.split()) < 40:
        indicators.append("Poor Description")

    if data_dict.get('has_company_logo') == 'No':
        indicators.append("No Company Logo")

    if data_dict.get('has_questions') == 'No':
        indicators.append("No Screening Questions")

    if data_dict.get('required_experience') in ['Unknown', None]:
        indicators.append("Missing Required Experience")

    if any(m in text_lower for m in OTHER_TERMS):
        indicators.append("Unrealistic Income Claims")

    if title_lower and any(t in title_lower for t in HIGH_RISK_TITLES):
        indicators.append("High-Risk Job Title")

    return indicators


    
def pre_process(text) :
  txt = text.lower()
  txt = re.sub(r'<.*?>', ' ' , txt)
  txt = re.sub(r'https?://\S+|www\.\S+', ' ', txt)
  txt = re.sub(r'\S+@\S+', ' ' , txt)
  txt = re.sub(r"[^a-z0-9\s.,!?;:'\"()\+-\[\]]"," ",txt)
  txt = re.sub(r'\s+',' ',txt).strip()
  return txt

with st.sidebar:
    st.markdown("### ⚙ AI Job Analyzer")
    page = st.radio(
        "Navigate", ["Home","Analyze Jobs","About"],
        label_visibility="collapsed",
    )

    st.divider()

    status = "🟢 Active" if model is not None else "🟡 Model Not Active"
    st.info(f"**Model:** {status}\n\n")


if page == 'Home' :

    st.title("🚀 Fake Job Postings Detector")
    st.subheader("Analyze and Verify Job Postings in Real-Time")

    st.divider()

    title = st.text_input("Enter Job Title")
    description = st.text_area("Job Description (Paste text here)")
    company_profile = st.text_area("Company Profile (Paste here)",value='Unknown')
    requirements = st.text_area("Requirements of Job Opening",value='Unknown')
    benefits = st.text_area("Enlisted offered benefits by the Employer",value='Unknown')

    col1, col2 = st.columns(2)
    with col1 :
        has_company_logo = st.selectbox("Has Company logo :",['Yes','No'])
        has_questions = st.selectbox("Has Company Screening Questions :",['Yes','No'])

    with col2 :
        employment_type = st.selectbox("Employment type :",['Other' ,'Full-time', 'Unknown', 'Part-time' ,'Contract', 'Temporary'],index=None,placeholder='Select Employment Type..') or 'Unknown'
        required_experience = st.selectbox("Required Experience :",['Internship' ,'Not Applicable' ,'Unknown', 'Mid-Senior level', 'Associate',
            'Entry level' ,'Executive', 'Director'],index=None,placeholder='Select Required Experience...') or 'Unknown'

    if st.button('Analyze Job Posting') :

        if vectorizer is None or model is None:
            st.error("Model pipeline is inactive.")

        else :
            my_dict = {
                'title' : title,
                'company_profile' : company_profile,
                'description' : description,
                'requirements' : requirements,
                'benefits' : benefits,
                'has_company_logo' : has_company_logo,
                'has_questions' : has_questions,
                'employment_type' : employment_type,
                'required_experience' : required_experience
            }

            # Extract Rule Indicators
            indicators = extract_indicators(my_dict)

            try :
                # Create Single-Row DataFrame
                df = pd.DataFrame([my_dict])

                df['combined_text'] = (
                    "[TITLE] " + df['title'] + " " +
                    "[COMPANY_PROFILE] " + df['company_profile'] + " " +
                    "[DESCRIPTION] " + df['description'] + " " +
                    "[REQUIREMENTS] " + df['requirements'] + " " +
                    "[BENEFITS] " + df['benefits']
                )

                df['combined_text'] = df['combined_text'].apply(pre_process)

                df['has_company_logo'].replace({'Yes' : 1,'No' : 0},inplace=True)
                df['has_questions'].replace({'Yes' : 1,'No' : 0},inplace=True)

                df.drop(columns=['title','company_profile','description','requirements','benefits'],inplace=True,axis=1)

                processed_data = vectorizer.transform(df)
                prediction = model.predict(processed_data)[0]
                
                prediction_scores = model.decision_function(processed_data)[0]
                probabilities = 1 / (1 + np.exp(-prediction_scores))
                score_percentage = int(probabilities * 100)
            
            except Exception as e:
                    st.error(f"An error occurred during feature processing: {e}")
            
            else :
                if prediction == 1 :
                    st.error("🔴 Fake Job Risk")
                else :
                    st.success("🟢 Likely Real Job")


                if score_percentage >= 70:
                    progress_color = "#FF4B4B"  # Red
                    status_icon = "⚠️"           # Warning Triangle
                elif 40 <= score_percentage < 70:   
                    progress_color = "#FFA500"  # Yellow
                    status_icon = "⚠️"           # Warning Triangle
                else:
                    progress_color = "#2E7D32"  # Green
                    status_icon = "✅"           # Green Checkmark
            
                # 3. Inject Custom CSS to dynamically color the progress bar
                st.markdown(
                    f"""
                    <style>
                        div[data-testid="stProgress"] > div > div > div > div {{
                            background-color: {progress_color} !important;
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True,  
                )
                
                # 4. Display the Confidence text
                st.write(f"Fake Job Probability: **{score_percentage}%**")
                
                # 5. Create two columns to put the icon and progress bar side-by-side
                # The first column is narrow for the icon, the second takes the rest of the space
                icon_col, bar_col = st.columns([1, 15])
                
                with icon_col:
                    # Use st.write to display the emoji icon cleanly aligned
                    st.write(status_icon)
                
                with bar_col:
                    # The progress bar renders right next to the icon
                    st.progress(probabilities)

                st.divider()
                
                if prediction == 1 :
                    st.markdown("Key Indicators of Fraud")

                    if indicators :
                        tags_html = "".join(f"<span class='fraud-tag'>{t}</span>" for t in indicators)
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.info("No fraud indicators detected.")


elif page == 'Analyze Jobs' :

    st.title("🎯Fake/Real Jobs Distributions")

    st.divider()


    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("1. Application Questions")
        st.image("visualizations/Has_questions.png", width=550) 

    with row1_col2:
        st.subheader("2. Company Logo Presence")
        st.image("visualizations/Has_Logo.png", width=550)

    st.divider()


    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("3. Fake Job Descriptions Type")
        st.image("visualizations/Fake_Descriptions.png", width=550)

    with row2_col2:
        st.subheader("4. Real Job Descriptions Type")
        st.image("visualizations/Non-Fake_Descriptions.png", width=550)

    st.divider()


    st.subheader("5. Historical Distribution of Required Experience with respect to Fraudulent Jobs")
    st.image("visualizations/Required_Experience.png", width=900, caption="Count of Distribution mapping")


elif page == "About" :

    st.title("ℹ️ About AI Job Analyzer")

    st.divider()

    st.write(
        """
        This tool uses a Support Vector Classifier (SVC) trained on TF-IDF
        features to flag potentially fraudulent job postings, based on
        patterns commonly seen in scam listings: urgency language,
        unrealistic pay claims, vague company details, and generic contact
        emails.

        """
    )

    st.divider()

    '''
        =====================================================Model's Metrics======================================================
     '''

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Accuracy","99%")
    col2.metric("Precision","93%")
    col3.metric("Recall","92%")
    col4.metric("PR_AUC","96%")

    st.info(""" 🎯The ultimate goal is to help job seekers make safer decisions,
    reduce exposure to employment scams, and demonstrate how artificial intelligence can be applied to
    address real-world cybersecurity and trust-related challenges in online recruitment.
     """)
