import streamlit as st
import os
import tempfile
import pytesseract
import logging
import time
from insurance_advisor import InsuranceAIAdvisor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Tesseract path (adjust for your system if needed)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
logging.info("Tesseract path set.")

# Initialize the advisor
@st.cache_resource
def init_advisor():
    logging.info("Initializing InsuranceAIAdvisor...")
    return InsuranceAIAdvisor()

try:
    advisor = init_advisor()
    logging.info("Advisor initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize advisor: {str(e)}")
    st.stop()

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 5px; }
    .stTextInput>div>input, .stTextArea textarea { border-radius: 5px; }
    .stFileUploader>div>div>input { border-radius: 5px; }
    .sidebar .sidebar-content { background-color: #e9ecef; }
    .stSuccess { background-color: #d4edda; }
    .stError { background-color: #f8d7da; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation and user ID
st.sidebar.title("Insurance AI Advisor")
user_id = st.sidebar.text_input("Enter User ID", value="guest_user")
page = st.sidebar.radio("Navigate", ["Home", "Query AI", "Upload Document", "View Documents", "User Profile"])

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}

# Header
st.title("Insurance AI Advisor")
st.markdown("Get personalized insurance advice for health, auto, and travel insurance.")

# Page: Home
if page == "Home":
    st.header("Welcome!")
    st.markdown("""
    This app helps you:
    - Upload and analyze insurance policy documents.
    - Get answers to insurance questions based on your documents.
    - View and manage your uploaded documents.
    - Update your profile for tailored recommendations.
    
    Use the sidebar to navigate or enter your User ID to start.
    """)

# Page: Query AI
elif page == "Query AI":
    st.header("Ask a Question")
    st.markdown("Ask about your insurance coverage, terms, or recommendations based on uploaded documents.")
    query = st.text_area("Your question:", placeholder="E.g., What is my deductible in my uploaded policy?", height=100)
    
    if st.button("Ask"):
        if query:
            with st.spinner("Processing your question..."):
                try:
                    response = advisor.query(user_id, query)
                    st.session_state.history.append({"query": query, "response": response})
                    st.markdown("**Response:**")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.error("Please enter a question.")

    # Display conversation history
    if st.session_state.history:
        st.subheader("Conversation History")
        for entry in st.session_state.history:
            with st.expander(f"You: {entry['query'][:50]}..."):
                st.markdown(f"**You:** {entry['query']}")
                st.markdown(f"**Advisor:** {entry['response']}")
                st.markdown("---")

    # Reset context button
    st.markdown("### Reset Conversation")
    reset_scope = st.radio("Reset scope:", ["Current User", "All Users"], horizontal=True)
    if st.button("Reset Now"):
        try:
            reset_user_id = user_id if reset_scope == "Current User" else "all"
            result = advisor.reset_context(reset_user_id)
            st.session_state.history = []
            if reset_user_id == "all":
                st.session_state.uploaded_docs = {}
            st.success(result)
        except Exception as e:
            st.error(f"Error resetting context: {str(e)}")

    # Example questions
    st.markdown("""
    **Example Questions:**
    - What is my health insurance deductible?
    - Does my auto policy cover rental cars?
    - What is covered under my travel insurance?
    """)

# Page: Upload Document
elif page == "Upload Document":
    st.header("Upload Insurance Document")
    st.markdown("Upload policy documents to get personalized answers. Supported formats: PDF, DOCX, DOC, TXT, JPG, JPEG, PNG")
    
    # Additional inputs for policy ID and document type
    policy_id = st.text_input("Policy ID (optional)", placeholder="E.g., ABC123")
    document_type = st.selectbox(
        "Document Type (optional)",
        ["", "health_policy", "auto_policy", "travel_policy", "general_policy"],
        index=0
    )
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        try:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                temp_path = os.path.join(temp_dir, file_name)
                
                # Check if file can be written
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                except Exception as e:
                    st.error(f"Error saving '{file_name}': {str(e)}")
                    continue
                
                # Validate file existence and size
                if not os.path.exists(temp_path):
                    st.error(f"Error: File '{file_name}' could not be saved.")
                    continue
                
                max_size_mb = advisor.config.MAX_DOCUMENT_SIZE / (1024 * 1024)
                if os.path.getsize(temp_path) > advisor.config.MAX_DOCUMENT_SIZE:
                    st.error(f"File '{file_name}' is too large. Maximum size is {max_size_mb}MB.")
                    continue
                
                # Process document
                with st.spinner(f"Processing '{file_name}'..."):
                    try:
                        result = advisor.upload_document(
                            file_path=temp_path,
                            file_name=file_name,
                            user_id=user_id,
                            policy_id=policy_id if policy_id else None,
                            document_type=document_type if document_type else None
                        )
                        
                        if result.get("status") == "success":
                            st.success(f"Successfully processed '{file_name}'")
                            st.markdown(f"**Policy ID:** {result.get('policy_number', 'Unknown')}")
                            st.markdown(f"**Document Type:** {result.get('document_type', 'Unknown')}")
                            st.session_state.uploaded_docs[file_name] = result
                        else:
                            st.error(f"Error processing '{file_name}': {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Processing '{file_name}' failed: {str(e)}")
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

# Page: View Documents
elif page == "View Documents":
    st.header("Your Documents")
    try:
        docs = advisor.get_user_documents(user_id)
        if not docs:
            st.info("No documents found. Upload a document to get started.")
        else:
            st.markdown(f"Found {len(docs)} documents:")
            for doc_id, metadata in docs.items():
                with st.expander(f"Document: {metadata.get('file_name', 'Unknown')}"):
                    st.markdown(f"**Policy ID:** {metadata.get('policy_id', 'Unknown')}")
                    st.markdown(f"**Type:** {metadata.get('document_type', 'Unknown')}")
                    st.markdown(f"**Uploaded:** {time.ctime(metadata.get('upload_time', 0))}")
                    if metadata.get("coverage_limits"):
                        st.markdown("**Coverage Limits:**")
                        for key, value in metadata["coverage_limits"].items():
                            st.markdown(f"- {key}: {value}")
                    if metadata.get("policy_period"):
                        st.markdown(f"**Policy Period:** {metadata['policy_period']['start_date']} to {metadata['policy_period']['end_date']}")
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")

# Page: User Profile
elif page == "User Profile":
    st.header("Manage Your Profile")
    try:
        profile = advisor.user_profile_manager.get_profile(user_id)
    except Exception as e:
        st.error(f"Error fetching profile: {str(e)}")
        profile = {}
    
    with st.form("profile_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=int(profile.get("age", 18)) or 18)
        occupation = st.text_input("Occupation", value=profile.get("occupation", ""))
        income = st.text_input("Income (e.g., $50k)", value=profile.get("income", ""))
        family_status = st.selectbox("Family Status", ["Single", "Married", "Divorced"], index=["Single", "Married", "Divorced"].index(profile.get("family_status", "Single")) or 0)
        medical_history = st.text_area("Medical History (comma-separated)", value=", ".join(profile.get("medical_history", [])))
        submit = st.form_submit_button("Update Profile")
        
        if submit:
            try:
                new_profile = {
                    "age": age,
                    "occupation": occupation,
                    "income": income,
                    "family_status": family_status,
                    "medical_history": [x.strip() for x in medical_history.split(",") if x.strip()]
                }
                advisor.update_user_profile(user_id, new_profile)
                st.success("Profile updated!")
            except Exception as e:
                st.error(f"Error updating profile: {str(e)}")
    
    st.subheader("Current Profile")
    if profile:
        for key, value in profile.items():
            st.markdown(f"**{key.capitalize()}:** {value}")
    else:
        st.info("No profile data found. Update your profile above.")

# Footer
st.markdown("---")
