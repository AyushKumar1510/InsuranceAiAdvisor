#!/usr/bin/env python
# coding: utf-8


"""
Advanced Insurance AI Advisor
Using Google's Gemini with RAG, ReAct, and Agent Architecture
"""

import os
import json
import google.generativeai as genai
from google.api_core.exceptions import TooManyRequests
from IPython.display import display, Markdown
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import logging
import io
import PyPDF2
from PIL import Image
import pytesseract
import docx
from google.api_core.exceptions import InvalidArgument
import time


# In[ ]:


# --- 1. Core Configuration ---

class Config:
    """Configuration settings for the Insurance AI Advisor"""
    API_KEY = "AIzaSyDFm8SK7-iyj7Vb0qhWF-1VnSMfDNWcZXY"
    VECTOR_DB_PATH = "./vector_db/"  # Local path for vector storage
    KNOWLEDGE_BASE_PATH = "./knowledge_base/"
    USER_PROFILES_PATH = "./user_profiles/"
    DOCUMENTS_PATH = "./documents/"  # Path for uploaded documents

    # Agent system settings
    VERBOSE = True  # Set to True for detailed agent reasoning
    MAX_STEPS = 5   # Maximum reasoning steps for ReAct

    # Document processing settings
    MAX_DOCUMENT_SIZE = 20 * 1024 * 1024  # 20MB limit for file uploads
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.jpg', '.jpeg', '.png']


# In[ ]:


# --- 2. API and Model Setup ---

def setup_models(config):
    """Initialize all required models"""
    # Set up Gemini
    genai.configure(api_key=config.API_KEY)
    llm = genai.GenerativeModel('models/gemini-1.5-flash')

    # Set up embedding model
    embedding_model = GoogleEmbeddingModel(config.API_KEY)

    return llm, embedding_model


# In[ ]:


# Create a wrapper for Google's embedding model
class GoogleEmbeddingModel:
    def __init__(self, api_key):
        import time # import the time module
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/embedding-001')

    def encode(self, texts):
        """Get embeddings for a list of texts"""
        if not texts:
            return []

        # Batch process to minimize API calls
        embeddings = []
        batch_size = 5

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []

            for text in batch:
                # Rate limiting
                time.sleep(0.5)  # Space out API calls
                try:
                    result = self.model.embed(text=text)
                    embedding = result["embedding"]
                    batch_embeddings.append(embedding)
                except Exception as e:
                    display(Markdown(f"Embedding error: {e}"))
                    # Create a zero vector as fallback
                    fallback_embedding = [0.0] * 768  # Typical embedding dimension
                    batch_embeddings.append(fallback_embedding)

            embeddings.extend(batch_embeddings)

        return np.array(embeddings)


# In[ ]:


# --- 3. Knowledge Base & Vector Database ---

class KnowledgeBase:
    """Insurance knowledge management and vector storage focused on health, auto, and travel"""

    def __init__(self, config, embedding_model):
        self.config = config
        self.embedding_model = embedding_model
        self.ensure_directories()
        self.load_knowledge()

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.config.VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(self.config.KNOWLEDGE_BASE_PATH, exist_ok=True)
        os.makedirs(self.config.USER_PROFILES_PATH, exist_ok=True)

    def load_knowledge(self):
        """Load or create the knowledge base"""
        # Check if we have a saved vector database
        vector_path = os.path.join(self.config.VECTOR_DB_PATH, "embeddings.npy")
        content_path = os.path.join(self.config.VECTOR_DB_PATH, "contents.json")

        if os.path.exists(vector_path) and os.path.exists(content_path):
            # Load existing database
            self.embeddings = np.load(vector_path)
            with open(content_path, 'r') as f:
                self.contents = json.load(f)
            logging.info(f"Loaded {len(self.contents)} knowledge entries")
        else:
            # Initialize and create basic knowledge
            self.contents = []
            self.create_initial_knowledge()
            self.compute_embeddings()
            self.save_database()
            logging.info("Created new knowledge base with embeddings")

    def create_initial_knowledge(self):
        """Create focused knowledge entries for health, auto, and travel insurance"""
        # Core insurance information
        insurance_core = [
            {
                "type": "core_info",
                "insurance_type": "health",
                "description": "Health insurance covers medical expenses for illnesses, injuries, and preventive care.",
                "key_components": ["Premiums", "Deductibles", "Copayments", "Network coverage", "Prescription coverage"],
                "required_info": "age, pre-existing conditions, preferred doctors/hospitals, prescription needs, budget, family size"
            },
            {
                "type": "core_info",
                "insurance_type": "auto",
                "description": "Auto insurance covers damage to vehicles and liability for injuries in accidents.",
                "key_components": ["Liability coverage", "Collision coverage", "Comprehensive coverage", "Uninsured motorist protection"],
                "required_info": "vehicle make/model/year, driving history, annual mileage, coverage level desired, number of drivers, location"
            },
            {
                "type": "core_info",
                "insurance_type": "travel",
                "description": "Travel insurance covers unexpected events during trips, including cancellations, medical emergencies, and lost luggage.",
                "key_components": ["Trip cancellation", "Medical coverage", "Emergency evacuation", "Baggage protection", "Travel delay"],
                "required_info": "destination, trip duration, trip cost, activities planned, pre-existing conditions, age of travelers"
            }
        ]

        # Essential insurance terminology
        insurance_terms = [
            {
                "type": "term",
                "term": "Deductible",
                "definition": "The amount you pay for covered services before your insurance plan starts to pay.",
                "example": "If your deductible is $1,000, you pay the first $1,000 of covered services yourself."
            },
            {
                "type": "term",
                "term": "Premium",
                "definition": "The amount you pay for your insurance every month or year.",
                "example": "If your premium is $500 per month, you'll pay $6,000 per year for your insurance."
            },
            {
                "type": "term",
                "term": "Copayment",
                "definition": "A fixed amount you pay for a covered service after you've paid your deductible.",
                "example": "If your copay for a doctor's visit is $20, you pay $20 when you see the doctor."
            },
            {
                "type": "term",
                "term": "Coinsurance",
                "definition": "The percentage of costs you pay for covered services after meeting your deductible.",
                "example": "With 20% coinsurance, if a service costs $100, you pay $20 and insurance pays $80."
            }
        ]

        # Question sets for interactive recommendations
        question_sets = [
            {
                "type": "question_set",
                "insurance_type": "health",
                "required_info": "age, pre-existing conditions, preferred doctors/hospitals, prescription needs, budget, family size"
            },
            {
                "type": "question_set",
                "insurance_type": "auto",
                "required_info": "vehicle make/model/year, driving history, annual mileage, coverage level desired, number of drivers, location"
            },
            {
                "type": "question_set",
                "insurance_type": "travel",
                "required_info": "destination, trip duration, trip cost, activities planned, pre-existing conditions, age of travelers"
            }
        ]

        # Common user profile types for recommendations
        profile_guidelines = [
            {
                "type": "guideline",
                "profile_type": "young_single",
                "description": "Young single adults with no dependents",
                "health_priorities": ["High-deductible plan with HSA", "Catastrophic coverage", "Preventive care"],
                "auto_priorities": ["Liability coverage", "Collision coverage", "Roadside assistance"],
                "travel_priorities": ["Basic medical coverage", "Trip cancellation", "Lost baggage"]
            },
            {
                "type": "guideline",
                "profile_type": "family_young_children",
                "description": "Families with young children",
                "health_priorities": ["Comprehensive family coverage", "Low deductible", "Pediatric services"],
                "auto_priorities": ["Higher liability limits", "Full coverage", "Multiple vehicle discount"],
                "travel_priorities": ["Family coverage", "Trip cancellation", "Emergency medical evacuation"]
            },
            {
                "type": "guideline",
                "profile_type": "senior",
                "description": "Adults over 65",
                "health_priorities": ["Medicare supplement", "Prescription coverage", "Long-term care"],
                "auto_priorities": ["Senior discount", "Limited mileage rates", "Roadside assistance"],
                "travel_priorities": ["Pre-existing condition coverage", "Trip cancellation", "Medical evacuation"]
            }
        ]

        # Common questions and answers
        common_questions = [
            {
                "type": "question",
                "insurance_type": "health",
                "question": "What's the difference between an HMO and PPO health plan?",
                "answer": "An HMO (Health Maintenance Organization) typically requires you to choose a primary care physician and get referrals to see specialists within the network. A PPO (Preferred Provider Organization) gives more flexibility to see providers outside the network, though at a higher cost, and usually doesn't require referrals."
            },
            {
                "type": "question",
                "insurance_type": "auto",
                "question": "Do I need comprehensive and collision coverage?",
                "answer": "If your vehicle is newer or of high value, comprehensive and collision coverage is recommended to protect against damage to your car. For older vehicles worth less than $4,000, you might consider dropping these coverages as repair costs could exceed the car's value."
            },
            {
                "type": "question",
                "insurance_type": "travel",
                "question": "Does travel insurance cover trip cancellation due to COVID-19?",
                "answer": "Many travel insurance policies now include coverage for trip cancellation if you contract COVID-19 before travel. However, general fear of traveling during a pandemic usually isn't covered. Some policies offer 'Cancel For Any Reason' coverage at an additional cost."
            }
        ]

        # Combine all knowledge
        self.contents = insurance_core + insurance_terms + question_sets + profile_guidelines + common_questions

    def compute_embeddings(self):
        """Compute embeddings for all content"""
        texts = []
        for item in self.contents:
            # Format text based on content type
            if item["type"] == "core_info":
                text = f"Insurance: {item['insurance_type']}. Description: {item['description']}. Components: {', '.join(item['key_components'])}. Required info: {item['required_info']}"
            elif item["type"] == "term":
                text = f"Term: {item['term']}. Definition: {item['definition']}. Example: {item['example']}"
            elif item["type"] == "question_set":
                text = f"Questions for {item['insurance_type']} insurance. Required info: {item['required_info']}"
            elif item["type"] == "guideline":
                health = f"Health priorities: {', '.join(item['health_priorities'])}" if "health_priorities" in item else ""
                auto = f"Auto priorities: {', '.join(item['auto_priorities'])}" if "auto_priorities" in item else ""
                travel = f"Travel priorities: {', '.join(item['travel_priorities'])}" if "travel_priorities" in item else ""
                text = f"Profile: {item['description']}. {health} {auto} {travel}"
            elif item["type"] == "question":
                text = f"{item['insurance_type']} question: {item['question']}. Answer: {item['answer']}"
            else:
                text = json.dumps(item)

            texts.append(text)

        # Compute embeddings
        self.embeddings = self.embedding_model.encode(texts)

    def save_database(self):
        """Save the current database to disk"""
        np.save(os.path.join(self.config.VECTOR_DB_PATH, "embeddings.npy"), self.embeddings)
        with open(os.path.join(self.config.VECTOR_DB_PATH, "contents.json"), 'w') as f:
            json.dump(self.contents, f)

    def add_entry(self, entry):
        """Add a new entry to the knowledge base"""
        self.contents.append(entry)
        self.compute_embeddings()  # Recompute all embeddings
        self.save_database()

    def search(self, query, top_k=3):
        """Search the knowledge base for relevant information"""
        query_embedding = self.embedding_model.encode([query])[0]

        # Calculate similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top matches
        indices = np.argsort(similarities)[::-1][:top_k]
        results = [self.contents[i] for i in indices]
        scores = [similarities[i] for i in indices]

        return results, scores


# --- Document Processing Module ---

class DocumentProcessor:
    """Handles processing of policy documents in various formats"""

    def __init__(self, config, embedding_model):
        self.config = config
        self.embedding_model = embedding_model
        self.documents_path = os.path.join(self.config.KNOWLEDGE_BASE_PATH, "documents")
        os.makedirs(self.documents_path, exist_ok=True)

        # Document metadata store
        self.metadata_path = os.path.join(self.documents_path, "metadata.json")
        self.load_metadata()

    def load_metadata(self):
        """Load document metadata from disk"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata for {len(self.metadata)} documents")
        else:
            self.metadata = {}
            logging.info("No existing document metadata found")

    def save_metadata(self):
        """Save document metadata to disk"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def process_file(self, file_path, file_name, user_id=None, policy_id=None, document_type=None):
        """Process a file and extract its textual content"""
        try:
            logging.info(f"Processing file: {file_name}")

            # Extract file extension
            ext = os.path.splitext(file_name)[1].lower()

            # Process based on file type
            if ext == '.pdf':
                text_content = self._extract_pdf_text(file_path)
            elif ext in ['.doc', '.docx']:
                text_content = self._extract_docx_text(file_path)
            elif ext in ['.jpg', '.jpeg', '.png']:
                text_content = self._extract_image_text(file_path)
            else:
                # For text files or other supported formats
                with open(file_path, 'r') as f:
                    text_content = f.read()

            # Create unique document ID
            doc_id = f"doc_{int(time.time())}_{os.path.basename(file_name)}"

            # Store the document
            document_path = os.path.join(self.documents_path, doc_id)
            with open(document_path, 'w') as f:
                f.write(text_content)

            # Extract key information using regex patterns for insurance documents
            policy_number = self._extract_policy_number(text_content)
            coverage_limits = self._extract_coverage_limits(text_content)
            policy_period = self._extract_policy_period(text_content)

            # Store metadata
            self.metadata[doc_id] = {
                "file_name": file_name,
                "document_type": document_type or self._guess_document_type(text_content),
                "policy_id": policy_id or policy_number,
                "user_id": user_id,
                "upload_time": time.time(),
                "coverage_limits": coverage_limits,
                "policy_period": policy_period,
                "chunks": self._chunk_document(text_content)
            }

            self.save_metadata()

            return {
                "doc_id": doc_id,
                "status": "success",
                "policy_number": policy_number,
                "document_type": self.metadata[doc_id]["document_type"]
            }

        except Exception as e:
            logging.info(f"Error processing file: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _extract_pdf_text(self, file_path):
        """Extract text from PDF files"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text

    def _extract_docx_text(self, file_path):
        """Extract text from DOCX files"""
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _extract_image_text(self, file_path):
        """Extract text from images using OCR"""
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    def _chunk_document(self, text, chunk_size=500, overlap=100):
        """Break document into chunks for better processing"""
        chunks = []
        start = 0

        # Clean the text a bit
        text = re.sub(r'\s+', ' ', text).strip()

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to find a good breakpoint (period or newline)
            if end < len(text):
                breakpoint = text.rfind('. ', start, end)
                if breakpoint != -1 and breakpoint > start + 100:  # Ensure chunk isn't too small
                    end = breakpoint + 1

            chunks.append({
                "text": text[start:end],
                "start": start,
                "end": end
            })

            start = end - overlap

        return chunks

    def _extract_policy_number(self, text):
        """Extract policy number using regex patterns common in insurance docs"""
        patterns = [
            r'Policy\s+Number[:\s]+([A-Z0-9-]+)',
            r'Policy\s+#[:\s]+([A-Z0-9-]+)',
            r'Policy\s+ID[:\s]+([A-Z0-9-]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_coverage_limits(self, text):
        """Extract coverage limits from document"""
        coverage_data = {}

        # Look for standard insurance coverage patterns
        coverage_patterns = [
            # Health insurance patterns
            (r'Deductible[:\s]+\$([\d,]+)', 'deductible'),
            (r'Out-of-pocket\s+maximum[:\s]+\$([\d,]+)', 'out_of_pocket_max'),
            (r'Coinsurance[:\s]+([\d]+)%', 'coinsurance_percentage'),

            # Auto insurance patterns
            (r'Bodily\s+Injury\s+Liability[:\s]+\$([\d,/]+)', 'bodily_injury'),
            (r'Property\s+Damage\s+Liability[:\s]+\$([\d,]+)', 'property_damage'),
            (r'Collision\s+Deductible[:\s]+\$([\d,]+)', 'collision_deductible'),

            # Travel insurance patterns
            (r'Trip\s+Cancellation[:\s]+\$([\d,]+)', 'trip_cancellation'),
            (r'Emergency\s+Medical[:\s]+\$([\d,]+)', 'emergency_medical')
        ]

        for pattern, key in coverage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                coverage_data[key] = match.group(1).replace(',', '')

        return coverage_data

    def _extract_policy_period(self, text):
        """Extract policy period dates"""
        pattern = r'Policy\s+Period:?\s+([\w\s,]+\d{1,2},\s+\d{4})\s+to\s+([\w\s,]+\d{1,2},\s+\d{4})'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return {
                "start_date": match.group(1).strip(),
                "end_date": match.group(2).strip()
            }

        return None

    def _guess_document_type(self, text):
        """Guess document type based on content"""
        text = text.lower()

        if re.search(r'health\s+insurance|medical\s+coverage|prescription', text):
            return "health_policy"
        elif re.search(r'auto\s+insurance|vehicle\s+coverage|collision|comprehensive', text):
            return "auto_policy"
        elif re.search(r'travel\s+insurance|trip\s+cancellation|baggage\s+loss', text):
            return "travel_policy"
        else:
            return "general_policy"

    def get_document_by_id(self, doc_id):
        """Retrieve a document by ID"""
        if doc_id not in self.metadata:
            return None

        doc_path = os.path.join(self.documents_path, doc_id)
        if not os.path.exists(doc_path):
            return None

        with open(doc_path, 'r') as f:
            content = f.read()

        return {
            "content": content,
            "metadata": self.metadata[doc_id]
        }

    def get_documents_by_user(self, user_id):
        """Get all documents for a specific user"""
        user_docs = {}
        for doc_id, metadata in self.metadata.items():
            if metadata.get("user_id") == user_id:
                user_docs[doc_id] = metadata
        return user_docs

    def get_all_documents(self):
        """Get all document metadata"""
        return self.metadata

    def search_documents(self, query, user_id=None, top_k=3):
        """Search documents using embeddings similarity"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        results = []

        # For each document
        for doc_id, metadata in self.metadata.items():
            # Skip if user_id is specified and doesn't match
            if user_id and metadata.get("user_id") != user_id:
                continue

            # Get document chunks
            chunks = metadata.get("chunks", [])

            for i, chunk in enumerate(chunks):
                # Get or create chunk embedding
                if "embedding" not in chunk:
                    # Rate limiting for API calls
                    time.sleep(0.5)
                    try:
                        chunk_embedding = self.embedding_model.encode([chunk["text"]])[0]
                        self.metadata[doc_id]["chunks"][i]["embedding"] = chunk_embedding.tolist()
                        self.save_metadata()
                    except Exception as e:
                        logging.info(f"Error generating embedding: {str(e)}")
                        continue
                else:
                    chunk_embedding = np.array(chunk["embedding"])

                # Calculate similarity
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )

                results.append({
                    "doc_id": doc_id,
                    "chunk_idx": i,
                    "text": chunk["text"],
                    "similarity": similarity,
                    "metadata": {
                        "file_name": metadata.get("file_name"),
                        "document_type": metadata.get("document_type"),
                        "policy_id": metadata.get("policy_id")
                    }
                })

        # Sort by similarity and return top_k
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
        return results



# --- 4. User Profile Management ---

class UserProfileManager:
    """Manages user profiles and preferences"""

    def __init__(self, config, embedding_model):
        self.config = config
        self.embedding_model = embedding_model
        self.profiles = {}
        self.load_profiles()

    def load_profiles(self):
        """Load saved user profiles"""
        profile_path = os.path.join(self.config.USER_PROFILES_PATH, "profiles.json")
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                self.profiles = json.load(f)
            logging.info(f"Loaded {len(self.profiles)} user profiles")
        else:
            logging.info("No existing profiles found")

    def save_profiles(self):
        """Save user profiles to disk"""
        with open(os.path.join(self.config.USER_PROFILES_PATH, "profiles.json"), 'w') as f:
            json.dump(self.profiles, f)

    def create_profile(self, user_id, profile_data):
        """Create or update a user profile"""
        self.profiles[user_id] = profile_data
        self.save_profiles()

    def get_profile(self, user_id):
        """Retrieve a user profile"""
        return self.profiles.get(user_id, {})


    def extract_profile_from_text(self, text, llm):
        """Use LLM to extract profile information from text with enhanced instructions"""
        prompt = f"""
        Extract structured user profile information from the following text.
        Focus on insurance-relevant details and be thorough.

        Return the information in JSON format with any of these fields that can be identified:

        BASIC INFORMATION:
        - age: integer
        - occupation: string
        - income: string (income range)
        - family_status: string (married, single, etc.)
        - number_of_dependents: integer

        HEALTH-RELATED:
        - medical_history: list of strings (conditions)
        - current_medications: list of strings

        AUTO-RELATED:
        - vehicle_type: string (make/model/year)
        - driving_history: string (accidents, violations)
        - annual_mileage: string


        TRAVEL-RELATED:
        - destination: string
        - trip_duration: string
        - trip_cost: string


        GENERAL:
        - assets: list of strings
        - liabilities: list of strings
        - risk_tolerance: string (Low, Medium, High)
        - existing_coverage: list of strings
        - priorities: list of strings

        Only include fields that can be confidently extracted from the text.

        Text: {text}
        """

        response = llm.generate_content(prompt)
        try:
            # Extract JSON from response
            match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = response.text

            profile = json.loads(json_str)
            return profile
        except:
            # Fallback if JSON parsing fails
            return {"raw_text": text}


# In[ ]:


# --- 5. RAG System ---

class RAGSystem:
    """Retrieval-Augmented Generation system for insurance knowledge"""

    def __init__(self, knowledge_base, document_processor=None):
        self.knowledge_base = knowledge_base
        self.document_processor = document_processor

    def query(self, user_query, user_profile=None, user_id = None, top_k=3):
        """Query the RAG system with user query and profile context"""
        # Expand query with profile if available
        expanded_query = user_query
        if user_profile:
            profile_summary = f"User is {user_profile.get('age', '')} years old, " + \
                             f"works as {user_profile.get('occupation', '')}, " + \
                             f"income level is {user_profile.get('income', '')}, " + \
                             f"family status is {user_profile.get('family_status', '')}"
            expanded_query = f"{profile_summary}. Query: {user_query}"

        # Get results from knowledge base
        kb_results, kb_scores = self.knowledge_base.search(expanded_query, top_k=top_k)

        # Get results from document processor if available
        doc_results = []
        if self.document_processor and user_id:
            doc_results = self.document_processor.search_documents(expanded_query, user_id=user_id, top_k=top_k)

        # Prepare context for LLM
        context = ""
        for i, (result, score) in enumerate(zip(kb_results, kb_scores)):
            if score < 0.5:  # Threshold for relevance
                continue

            if result["type"] == "term":
                context += f"TERM: {result['term']}\nDEFINITION: {result['definition']}\nEXAMPLE: {result['example']}\n\n"
            elif result["type"] == "policy":
                context += f"POLICY: {result['name']}\nDESCRIPTION: {result['description']}\nBENEFITS: {', '.join(result['benefits'])}\nCONSIDERATIONS: {', '.join(result['considerations'])}\n\n"
            elif result["type"] == "guideline":
                context += f"PROFILE GUIDELINE: {result['description']}\nPRIORITIES: {', '.join(result['priorities'])}\nCONSIDERATIONS: {', '.join(result['considerations'])}\n\n"
            elif result["type"] == "question":
                context += f"Q: {result['question']}\nA: {result['answer']}\n\n"
         # Add document context
        if doc_results:
            context += "POLICY DOCUMENT INFORMATION:\n"
            for i, result in enumerate(doc_results):
                if result["similarity"] < 0.5:  # Threshold for relevance
                    continue

                context += f"FROM DOCUMENT: {result['metadata']['file_name']} (Policy ID: {result['metadata']['policy_id'] or 'Unknown'})\n"
                context += f"CONTEXT: {result['text']}\n\n"
        return context

    def augment_prompt(self, user_query, user_profile=None, top_k=3):
        """Create an augmented prompt with retrieved context"""
        context = self.query(user_query, user_profile, top_k)

        # Create the augmented prompt
        if context:
            augmented_prompt = f"""
            Use the following information to help answer the user's question:

            REFERENCE INFORMATION:
            {context}

            USER QUERY: {user_query}

            Provide a helpful, accurate response based on the reference information and your knowledge.
            If the reference information is not relevant to the query, rely on your general knowledge about insurance.
            Always prioritize being helpful and accurate.
            If specific policy information is available in the reference, refer to it directly in your answer.
            """
        else:
            augmented_prompt = f"""
            Answer the following insurance-related question based on your knowledge:

            USER QUERY: {user_query}

            Provide a helpful, accurate response about insurance concepts, coverage options, or recommendations.
            """

        return augmented_prompt


# In[ ]:


# --- 6. Agent System with ReAct ---

class InsuranceAgent:
    """Base class for insurance-specific agents using ReAct paradigm"""

    def __init__(self, name, description, llm, rag_system, config):
        self.name = name
        self.description = description
        self.llm = llm
        self.rag_system = rag_system
        self.config = config

    def react(self, user_query, user_profile=None):
        """Run the ReAct reasoning process"""
        # Create the initial prompt
        prompt = self._create_react_prompt(user_query, user_profile)

        # Run the ReAct loop
        thoughts = []
        for step in range(self.config.MAX_STEPS):
            # Get response from LLM - no need for retry logic as Orchestrator handles this
            response = self.llm.generate_content(prompt)

            # Parse the response
            thought, action, action_input = self._parse_react_response(response.text)
            thoughts.append({"thought": thought, "action": action, "action_input": action_input})

            # If verbose, show the reasoning
            if self.config.VERBOSE:
                logging.info(f"--- Step {step+1} ---")
                logging.info(f"Thought: {thought}")
                logging.info(f"Action: {action}")
                logging.info(f"Action Input: {action_input}")

            # If final answer, apply grounding and return it
            if action == "Final Answer":
                grounded_response = self._ground_response(action_input)
                return grounded_response

            # Otherwise, perform the action and update the prompt
            action_result = self._perform_action(action, action_input)

            # Add to prompt
            prompt += f"\nObservation: {action_result}\nThought:"

        # If we've reached the maximum steps, return the last thought
        return f"I've been thinking about this problem but need more information. Here's what I know so far: {thoughts[-1]['thought']}"

    def _create_react_prompt(self, user_query, user_profile):
        """Create the initial ReAct prompt"""
        # Create a profile summary if available
        profile_summary = ""
        if user_profile:
            profile_items = []
            for key, value in user_profile.items():
                if value and key != "raw_text":
                    if isinstance(value, list):
                        profile_items.append(f"{key}: {', '.join(value)}")
                    else:
                        profile_items.append(f"{key}: {value}")

            if profile_items:
                profile_summary = "User Profile:\n" + "\n".join(profile_items)

        # Basic ReAct prompt template - focus on health, auto, and travel only
        prompt = f"""
        You are {self.name}, {self.description}.

        Your task is to answer this insurance-related query:
        {user_query}

        {profile_summary}

        Think through this step-by-step:

        1. First, understand what information you need
        2. Use available actions to gather that information
        3. Analyze the information
        4. Provide a final answer

        Available Actions:
        - Search Insurance Knowledge: Search for specific insurance terms, policies, or concepts
        - Analyze User Profile: Analyze the user's profile to understand their needs
        - Compare Options: Compare different insurance options based on criteria
        - Explain Term: Provide a detailed explanation of an insurance term
        - Final Answer: Provide your final recommendation or answer

        IMPORTANT: Only focus on health, auto, and travel insurance. Do not provide information about home or life insurance.

        Follow this format:

        Thought: I need to think about what information I need to answer this query
        Action: [one of the available actions]
        Action Input: [input for the action]

        Start now:

        Thought:
        """

        return prompt

    def _parse_react_response(self, response_text):
        """Parse the ReAct response from the LLM"""
        thought_match = re.search(r'Thought:(.*?)(?:Action:|$)', response_text, re.DOTALL)
        action_match = re.search(r'Action:(.*?)(?:Action Input:|$)', response_text, re.DOTALL)
        action_input_match = re.search(r'Action Input:(.*?)(?:Observation:|$)', response_text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else "Final Answer"
        action_input = action_input_match.group(1).strip() if action_input_match else ""

        # If no explicit action but there's a thought, treat it as the final answer
        if not action_match and thought:
            action = "Final Answer"
            action_input = thought

        return thought, action, action_input

    def _perform_action(self, action, action_input):
        """Perform the specified action and return the result"""
        if action == "Search Insurance Knowledge":
            # Search the knowledge base
            results, scores = self.rag_system.knowledge_base.search(action_input, top_k=2)
            result_texts = []
            for result, score in zip(results, scores):
                if score < 0.5:  # Relevance threshold
                    continue

                if result["type"] == "term":
                    result_texts.append(f"Term: {result['term']} - {result['definition']}")
                elif result["type"] == "policy":
                    result_texts.append(f"Policy: {result['name']} - {result['description']}")
                elif result["type"] == "guideline":
                    result_texts.append(f"Guideline: {result['description']}")
                elif result["type"] == "question":
                    result_texts.append(f"Q: {result['question']} - A: {result['answer']}")

            if result_texts:
                return "\n".join(result_texts)
            else:
                return "No relevant information found in knowledge base."

        elif action == "Analyze User Profile":
            # Analyze the user profile for insurance needs
            profile_prompt = f"""
            Analyze this user profile information for insurance needs related to health, auto, and travel insurance:
            {action_input}

            Identify:
            1. Primary insurance needs based on life stage
            2. Risk factors to consider
            3. Coverage priorities

            Provide a brief analysis focusing only on health, auto, and travel insurance.
            """

            response = self.llm.generate_content(profile_prompt)
            return response.text

        elif action == "Compare Options":
            # Compare insurance options
            compare_prompt = f"""
            Compare these insurance options (focusing only on health, auto, and travel insurance):
            {action_input}

            Provide a brief comparison of pros and cons.
            """

            response = self.llm.generate_content(compare_prompt)
            return response.text

        elif action == "Explain Term":
            # Explain an insurance term
            term_prompt = f"""
            Explain this insurance term in simple language:
            {action_input}

            Include a definition and practical example, preferably related to health, auto, or travel insurance.
            """

            response = self.llm.generate_content(prompt)
            return response.text

        else:
            return "Action not recognized. Available actions: Search Insurance Knowledge, Analyze User Profile, Compare Options, Explain Term, Final Answer"

    def _ground_response(self, response):
        """Add grounding to responses with sources and evidence"""
        grounding_prompt = f"""
        Review this insurance recommendation/explanation and add appropriate grounding:

        {response}

        Add 3-5 specific source references that support key claims in the response.
        Focus on authoritative insurance sources or principles related to health, auto, and travel insurance.
        Format these as a "Sources:" section at the end of the response.
        Do not invent sources - use general but accurate references.
        """

        try:
            grounding_result = self.llm.generate_content(grounding_prompt)
            return grounding_result.text
        except Exception as e:
            logging.info(f"Grounding error: {e}")
            # If grounding fails, return the original response
            return response


# In[ ]:


# --- 7. Specialized Agents ---

class RecommendationAgent(InsuranceAgent):
    """Agent specialized in recommending insurance policies"""

    def __init__(self, llm, rag_system, config):
        super().__init__(
            "Insurance Recommendation Specialist",
            "an expert in analyzing user needs and recommending appropriate insurance coverage",
            llm, rag_system, config
        )

    def _create_react_prompt(self, user_query, user_profile):
        base_prompt = super()._create_react_prompt(user_query, user_profile)

        # Add recommendation-specific instructions
        specialized_prompt = base_prompt + """

        For recommendations, consider:
        1. Life stage and family situation
        2. Income level and financial obligations
        3. Asset protection needs
        4. Risk tolerance
        5. Existing coverage gaps

        Your final recommendation should include:
        - Priority insurance types with coverage levels
        - Explanation of why these are appropriate
        - Estimated cost ranges (if applicable)
        - Additional considerations
        """

        return specialized_prompt

class ExplainerAgent(InsuranceAgent):
    """Agent specialized in explaining insurance concepts"""

    def __init__(self, llm, rag_system, config):
        super().__init__(
            "Insurance Terminology Expert",
            "an expert in explaining complex insurance concepts in simple terms",
            llm, rag_system, config
        )

    def _create_react_prompt(self, user_query, user_profile):
        base_prompt = super()._create_react_prompt(user_query, user_profile)

        # Add explanation-specific instructions
        specialized_prompt = base_prompt + """

        For explanations, make sure to:
        1. Start with a simple definition
        2. Provide real-world examples
        3. Explain why this matters to the policyholder
        4. Address common misconceptions
        5. Use analogies where helpful

        Avoid insurance jargon when possible, or explain the jargon when used.
        """

        return specialized_prompt


# In[ ]:


# --- 8. Orchestrator ---


class AgentOrchestrator:
    """Orchestrates the different agents based on user intent with minimal API calls"""

    def __init__(self, llm, rag_system, user_profile_manager, config):
        self.llm = llm
        self.rag_system = rag_system
        self.user_profile_manager = user_profile_manager
        self.config = config

        # Initialize agents (only create when needed to reduce overhead)
        self.recommendation_agent = None
        self.explainer_agent = None

        # Add response caching to reduce API calls
        self.response_cache = {}

        # Last API call timestamp for rate limiting
        self.last_api_call = 0

        # Track conversation history
        self.conversation_history = []

        # Initialize session tracking data
        self._last_insurance_type = {}
        self._last_query_context = {}

    def _rate_limited_api_call(self, prompt):
        """Make an API call with rate limiting to avoid quota issues"""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call

        # Enforce at least 2 seconds between API calls
        if time_since_last < 2:
            sleep_time = 2 - time_since_last
            logging.info(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

        # Try the API call with retries
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                self.last_api_call = time.time()
                return response
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        # On last attempt, return a fallback response
                        return type('obj', (object,), {'text': "I'm currently experiencing high demand. Please try again in a few moments."})
                else:
                    print(f"API error: {str(e)}")
                    return type('obj', (object,), {'text': f"I encountered an error: {str(e)}"})

    def _looks_like_new_topic(self, query):
        """Enhanced detection if user is explicitly starting a new topic"""
        new_topic_indicators = [
            "new query", "different question", "another topic",
            "not related", "this is not", "different subject",
            "new topic", "change topic", "switch topic",
            "this is new", "new conversation"
        ]

        query_lower = query.lower()
        # More aggressive detection of new topics
        return (
            any(indicator in query_lower for indicator in new_topic_indicators) or
            query_lower.startswith("new") or
            query_lower == "reset"
        )

    def reset_conversation_context(self, user_id="all"):
        """Completely reset the conversation context and all associated data when needed"""
        print(f"Performing complete reset for user: {user_id}")

        if user_id == "all":
            # Reset all conversations
            print("Resetting all conversation contexts and user data")
            self.conversation_history = []
            self.response_cache = {}

            # Also clear any session-level data that might be persisting
            self._clear_session_data()
        else:
            # Reset just this user's conversation
            logging.info(f"Resetting conversation context for user {user_id}")
            # Filter out this user's messages
            self.conversation_history = [msg for msg in self.conversation_history
                                        if not (msg.get("user_id") == user_id)]
            # Clear cache for this user
            keys_to_remove = [k for k in self.response_cache.keys()
                            if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                self.response_cache.pop(key, None)

            # Also clear user-specific session data
            self._clear_session_data(user_id)

        return "## Conversation Reset\n\nYour conversation has been reset. What would you like to know about health, auto, or travel insurance?"

    def _clear_session_data(self, user_id=None):
        """Clear any session-level data that might be persisting between conversations"""
        # Reset last detected insurance type
        if user_id is None:  # Reset for all users
            self._last_insurance_type = {}
            self._last_query_context = {}
        else:  # Reset for specific user
            if user_id in self._last_insurance_type:
                del self._last_insurance_type[user_id]
            if user_id in self._last_query_context:
                del self._last_query_context[user_id]

    def process_query(self, user_id, user_query):
        """Process the user query with improved context handling"""
        try:
            logging.info(f"Processing query: {user_query}")

            # Check if user explicitly wants to start a new topic or reset
            if self._looks_like_new_topic(user_query):
                logging.info("New topic detected! Performing context reset.")
                # Perform a complete reset for this user
                self.reset_conversation_context(user_id)
                # If it's just the word "reset", return the reset message
                if user_query.lower().strip() == "reset":
                    return "## Conversation Reset\n\nYour conversation has been reset. What would you like to know about health, auto, or travel insurance?"

            # Add to conversation history - with user_id attached for tracking
            self.conversation_history.append({"role": "user", "content": user_query, "user_id": user_id})

            # Check cache first with a more specific key that includes context length
            cache_key = f"{user_id}:{user_query}:{len(self.conversation_history)}"
            if cache_key in self.response_cache:
                logging.info("Using cached response")
                cached_response = self.response_cache[cache_key]
                self.conversation_history.append({"role": "assistant", "content": cached_response, "user_id": user_id})
                return cached_response

            # Detect current insurance type
            insurance_type = self._detect_insurance_type(user_query)

            # Store the current detected type
            self._last_insurance_type[user_id] = insurance_type

            # If this looks like profile information and type is "general",
            # use the previous context instead
            if insurance_type == "general" and self._looks_like_profile_info(user_query):
                if len(self.conversation_history) >= 2:
                    for msg in reversed(self.conversation_history):
                        if msg.get("role") == "user" and msg.get("user_id") == user_id:
                            previous_type = self._detect_insurance_type(msg.get("content", ""))
                            if previous_type != "general":
                                logging.info(f"Profile information detected. Using previous context: {previous_type} insurance")
                                insurance_type = previous_type
                                # Update the stored type
                                self._last_insurance_type[user_id] = insurance_type
                                break

            print(f"Current query is about: {insurance_type} insurance")

            # Check if topic has changed
            previous_type = self._last_insurance_type.get(user_id, "general")
            if (previous_type != insurance_type and
                previous_type != "general" and
                insurance_type != "general" and
                len(self.conversation_history) > 1):
                logging.info(f"Topic changed from {previous_type} to {insurance_type}. Performing partial reset.")
                # Store the current context in session data
                self._last_query_context[user_id] = {
                    "insurance_type": insurance_type,
                    "query": user_query
                }
                # We don't want to completely reset, but we want to make sure the model knows we've changed topics
                self.conversation_history.append({"role": "system", "content": f"TOPIC_CHANGE: Now discussing {insurance_type} insurance"})

            # Check if this is a recommendation query
            is_recommendation = self._is_recommendation_query(user_query)
            logging.info(f"Is recommendation query: {is_recommendation}")

            # Get user profile
            user_profile = self.user_profile_manager.get_profile(user_id)

            # Check if query is specifically about policy documents
            is_policy_specific = self._is_policy_specific_query(user_query)
            logging.info(f"Is policy-specific query: {is_policy_specific}")

            # For recommendations, check if we need follow-up questions
            if is_recommendation and insurance_type in ["health", "auto", "travel"]:
                has_sufficient_info = self._check_profile_completeness(user_profile, insurance_type)

                if not has_sufficient_info:
                    # Use a simplified single API call approach for follow-up questions
                    follow_up_response = self._simplified_follow_up_questions(user_query, insurance_type, user_profile)
                    self.conversation_history.append({"role": "assistant", "content": follow_up_response, "user_id": user_id})

                    # Cache the response
                    self.response_cache[cache_key] = follow_up_response
                    return follow_up_response

            # For all other cases, use a simplified single API call approach
            response = self._simplified_query_processing(user_id, user_query, insurance_type, user_profile)

            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response, "user_id": user_id})

            # Cache the response
            self.response_cache[cache_key] = response
            return response

        except Exception as e:
            logging.info(f"Error in process_query: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def _simplified_follow_up_questions(self, user_query, insurance_type, user_profile):
        """Generate follow-up questions with a single API call"""
        # Get required info for this insurance type
        required_info = self._get_required_info_for_type(insurance_type)

        # Create a prompt for generating follow-up questions
        prompt = f"""
        You are an interactive insurance advisor specializing in {insurance_type} insurance.

        USER QUERY: {user_query}

        CURRENT USER INFORMATION:
        {json.dumps(user_profile) if user_profile else "No information available yet."}

        IMPORTANT CONTEXT INSTRUCTION:
        Focus ONLY on {insurance_type} insurance for this query.
        Ignore any previous conversation about other insurance types.

        For {insurance_type} insurance recommendations, you typically need information about: {required_info}

        INSTRUCTIONS:
        1. Review what information the user has already provided.
        2. Ask 2-3 specific, focused questions to gather the most critical missing information needed for a {insurance_type} insurance recommendation.
        3. Be conversational and explain briefly why you're asking these questions.
        4. Focus only on the most important missing information.
        5. Format your questions clearly so they're easy to answer.
        6. Briefly reference 1-2 insurance industry practices or standards that explain why this information is important.

        Do not provide recommendations yet - just ask the follow-up questions to gather information.
        """

        # Make rate-limited API call
        response = self._rate_limited_api_call(prompt)
        return response.text

    def _simplified_query_processing(self, user_id, user_query, insurance_type, user_profile):
        """Process queries with a single API call instead of using agents"""
        # Extract any profile information from the query (without API call)
        extracted_profile = self._extract_basic_profile_info(user_query)

        # Update profile if new info found
        if extracted_profile:
            if user_profile:
                # Merge profiles
                for key, value in extracted_profile.items():
                    if value:  # Only update if not empty
                        user_profile[key] = value
            else:
                user_profile = extracted_profile

            # Save updated profile
            self.user_profile_manager.create_profile(user_id, user_profile)

        # Check if query is about a specific policy document
        is_policy_specific = self._is_policy_specific_query(user_query)

        # Create a comprehensive context from RAG system without API call
        # If policy specific, include user_id to search their documents
        if is_policy_specific:
            context = self.rag_system.query(user_query, user_profile, user_id)
        else:
            context = self._get_relevant_knowledge(user_query, insurance_type)

        # Create a unified prompt that handles all query types with strong focus on current topic only
        prompt = f"""
        You are an insurance advisor specializing in {insurance_type} insurance.

        CURRENT USER QUERY: {user_query}

        USER PROFILE:
        {json.dumps(user_profile) if user_profile else "No profile information available yet."}

        RELEVANT INSURANCE KNOWLEDGE:
        {context}

        CRITICAL CONTEXT INSTRUCTION:
        Focus EXCLUSIVELY on {insurance_type} insurance for this specific query.
        You must NOT reference or consider any previous conversations about other types of insurance.
        Answer as if this is a completely new conversation specifically about {insurance_type} insurance.

        Do not assume the user has any prior context or is referring to any previous conversations
        unless explicitly stated in their current query.

        If the user mentions they are changing topics, starting a new conversation, or asking something unrelated
        to previous exchanges, treat this as a completely fresh conversation with no connection to prior discussions.

        Based on the user's query:

        1. If they want an explanation of an insurance term, provide a clear, simple explanation with examples.
        2. If they want a recommendation, provide a detailed recommendation based on their profile information.
        3. If they want to compare options, explain the key differences between the options they're considering.
        4. If they have a claims question, explain the typical process and considerations.
        5. If they're asking about specific policy information and relevant policy documents were found, refer directly to those documents in your answer.
        6. For any other query, provide a helpful, accurate response based on insurance best practices.

        IMPORTANT: Only provide information about health, auto, and travel insurance. Do not offer recommendations
        or information about home insurance or life insurance.

        GROUNDING REQUIREMENTS:
        - Add 2-3 specific sources or evidence that support your key points
        - Include a "Sources:" section at the end of your response
        - Sources should be authoritative and specific to insurance (e.g., "According to the National Association of Insurance Commissioners...")
        - For insurance principles, cite commonly accepted industry practices
        - If you're referring to information from the user's policy documents, clearly indicate this

        Format your response in a conversational but professional tone.
        """

        # Make rate-limited API call
        response = self._rate_limited_api_call(prompt)
        return response.text

    def _is_policy_specific_query(self, query):
        """Determine if query is about a specific policy document"""
        policy_indicators = [
            "my policy", "in my policy", "policy document", "policy terms",
            "coverage details", "my plan", "what does my policy", "terms and conditions",
            "what am I covered for", "what's covered", "in the document", "doc says",
            "coverage limit", "according to my policy"
        ]

        # Check if any indicator is in the query
        return any(indicator.lower() in query.lower() for indicator in policy_indicators)

    def _get_relevant_knowledge(self, query, insurance_type):
        """Get relevant knowledge without API calls"""
        try:
            # Simple keyword-based knowledge retrieval (no embeddings/API calls)
            relevant_entries = []

            # Only consider health, auto, and travel insurance
            if insurance_type not in ["health", "auto", "travel", "general"]:
                insurance_type = "general"

            # First, look for insurance type specific entries
            type_entries = [entry for entry in self.rag_system.knowledge_base.contents
                           if isinstance(entry, dict) and
                           entry.get("insurance_type", "") == insurance_type]
            if type_entries:
                relevant_entries.extend(type_entries[:2])  # Add up to 2 type-specific entries

            # Then, look for term definitions if it's an explanation query
            if "what is" in query.lower() or "explain" in query.lower() or "definition" in query.lower():
                term_entries = [entry for entry in self.rag_system.knowledge_base.contents
                               if isinstance(entry, dict) and
                               entry.get("type", "") == "term"]

                # Try to find the specific term
                query_words = set(query.lower().split())
                for entry in term_entries:
                    term = entry.get("term", "").lower()
                    if term in query_words:
                        relevant_entries.append(entry)
                        break

            # Format the relevant entries
            context = ""
            for entry in relevant_entries:
                if entry.get("type") == "term":
                    context += f"TERM: {entry.get('term')}\nDEFINITION: {entry.get('definition')}\nEXAMPLE: {entry.get('example')}\n\n"
                elif entry.get("type") == "core_info":
                    context += f"INSURANCE TYPE: {entry.get('insurance_type')}\nDESCRIPTION: {entry.get('description')}\nKEY COMPONENTS: {', '.join(entry.get('key_components', []))}\n\n"
                else:
                    context += json.dumps(entry) + "\n\n"

            return context
        except Exception as e:
            logging.info(f"Error in _get_relevant_knowledge: {str(e)}")
            return "No relevant knowledge found."

    def _extract_basic_profile_info(self, text):
        """Extract basic profile info without API calls using regex"""
        profile = {}
        text = text.lower()

        # Age extraction
        age_match = re.search(r"(?:i am|i'm)\s+(\d+)(?:\s+years?\s+old)?", text)
        if age_match:
            profile["age"] = age_match.group(1)

        # Family status
        if "married" in text:
            profile["family_status"] = "married"
        elif "single" in text:
            profile["family_status"] = "single"

        # Children/dependents
        children_match = re.search(r"(\d+)\s+(?:kids|children)", text)
        if children_match:
            profile["children"] = children_match.group(1)

        # Income extraction
        income_match = re.search(r"\$(\d+)k|\$(\d+),000|\$(\d+) thousand", text)
        if income_match:
            groups = income_match.groups()
            if groups[0]:  # $90k format
                profile["income"] = f"${groups[0]}000"
            elif groups[1]:  # $90,000 format
                profile["income"] = f"${groups[1]}000"
            elif groups[2]:  # $90 thousand format
                profile["income"] = f"${groups[2]}000"

        # Travel-specific information
        if "travel" in text or "trip" in text or "vacation" in text:
            # Destination
            countries = ["usa", "canada", "mexico", "europe", "asia", "australia", "africa",
                         "france", "italy", "spain", "germany", "uk", "japan", "china", "thailand"]
            for country in countries:
                if country in text:
                    profile["destination"] = country
                    break

            # Trip duration
            duration_match = re.search(r"(\d+)\s+(?:days?|weeks?|months?)", text)
            if duration_match:
                value = duration_match.group(1)
                unit = re.search(r"\d+\s+(days?|weeks?|months?)", text).group(1)
                profile["trip_duration"] = f"{value} {unit}"

        # Auto-specific information
        if "car" in text or "vehicle" in text or "drive" in text:
            # Vehicle type
            car_brands = ["toyota", "honda", "ford", "chevrolet", "bmw", "mercedes", "audi", "volkswagen"]
            for brand in car_brands:
                if brand in text:
                    profile["vehicle_type"] = brand
                    break

        # Health-specific information
        if "health" in text or "medical" in text:
            # Medical conditions
            conditions = ["diabetes", "asthma", "high blood pressure", "hypertension", "heart disease"]
            found_conditions = []
            for condition in conditions:
                if condition in text:
                    found_conditions.append(condition)
            if found_conditions:
                profile["medical_history"] = found_conditions

        return profile

    def _get_required_info_for_type(self, insurance_type):
        """Get required information fields for an insurance type"""
        required_info_map = {
            "health": "age, pre-existing conditions, preferred doctors/hospitals, prescription needs, budget, family size",
            "auto": "vehicle make/model/year, driving history, annual mileage, coverage level desired, number of drivers, location",
            "travel": "destination, trip duration, trip cost, activities planned, pre-existing conditions, age of travelers"
        }

        return required_info_map.get(insurance_type, "personal information and insurance needs")

    def _detect_insurance_type(self, query):
        """Detect which type of insurance the user is asking about - limited to health, auto, travel"""
        query = query.lower()

        if any(term in query for term in ["health insurance", "medical insurance", "healthcare"]):
            return "health"
        elif any(term in query for term in ["auto insurance", "car insurance", "vehicle insurance", "motor insurance"]):
            return "auto"
        elif any(term in query for term in ["travel insurance", "trip insurance", "vacation insurance"]):
            return "travel"
        else:
            return "general"

    def _is_recommendation_query(self, query):
        """Check if this is a recommendation query"""
        recommendation_terms = ["recommend", "suggestion", "best for me", "should i get",
                               "what insurance", "need insurance", "looking for insurance", "best"]
        return any(term in query.lower() for term in recommendation_terms)

    def _check_profile_completeness(self, profile, insurance_type):
        """Check if the user profile has sufficient information for recommendations"""
        # Define minimum required fields for each insurance type - only for health, auto, travel
        required_fields = {
            "health": ["age", "medical_history"],
            "auto": ["vehicle_type", "driving_history"],
            "travel": ["destination", "trip_duration"]
        }

        # Get required fields for this insurance type
        fields_needed = required_fields.get(insurance_type, [])

        # Check if profile has the required fields
        if not profile:
            return False

        return all(field in profile for field in fields_needed)

    def _looks_like_profile_info(self, query):
        """Check if a query looks like profile information rather than a direct question"""
        profile_indicators = [
            "age", "years old", "condition", "budget", "coverage", "euro", "dollar",
            "medication", "doctor", "preferred", "pre-existing", "individual", "family"
        ]

        # Count how many profile indicators are present
        count = sum(1 for indicator in profile_indicators if indicator in query.lower())

        # If several indicators are present, it's likely profile information
        return count >= 2

    # Legacy methods preserved for compatibility but not used in the optimized flow
    def classify_intent(self, user_query):
        """Legacy method - not used in optimized flow"""
        return "general"  # Default to avoid unnecessary API calls

    def extract_profile_info(self, user_query):
        """Legacy method - use _extract_basic_profile_info instead"""
        return self._extract_basic_profile_info(user_query)



# --- 9. Main Interface ---

class InsuranceAIAdvisor:
    """Main interface for the Insurance AI Advisor"""

    def __init__(self):
        # Initialize configuration
        self.config = Config()

        # Setup models
        self.llm, self.embedding_model = setup_models(self.config)

        # Initialize knowledge system
        self.knowledge_base = KnowledgeBase(self.config, self.embedding_model)

        # Initialize document processor
        self.document_processor = DocumentProcessor(self.config, self.embedding_model)

        self.rag_system = RAGSystem(self.knowledge_base,self.document_processor)

        # Initialize user profile manager
        self.user_profile_manager = UserProfileManager(self.config, self.embedding_model)

        # Initialize orchestrator
        self.orchestrator = AgentOrchestrator(self.llm, self.rag_system, self.user_profile_manager, self.config)

        print("Insurance AI Advisor initialized with document handling capabilities")

    def _detect_insurance_type(self, query):
          """Detect which type of insurance the user is asking about"""
          query = query.lower()

          if any(term in query for term in ["health insurance", "medical insurance", "healthcare"]):
              return "health"
          elif any(term in query for term in ["auto insurance", "car insurance", "vehicle insurance"]):
              return "auto"
          elif any(term in query for term in ["travel insurance", "trip insurance", "vacation insurance"]):
              return "travel"
          elif any(term in query for term in ["home insurance", "homeowners insurance", "property insurance"]):
              return "home"
          elif any(term in query for term in ["life insurance", "death benefit"]):
              return "life"
          else:
              return "general"

    def query(self, user_id, user_query):
        """Process a user query"""
        return self.orchestrator.process_query(user_id, user_query)

    def add_knowledge(self, entry):
        """Add entry to knowledge base"""
        self.knowledge_base.add_entry(entry)

    def update_user_profile(self, user_id, profile_data):
        """Update a user's profile"""
        self.user_profile_manager.create_profile(user_id, profile_data)

    def reset_context(self, user_id="all"):
        """Reset the conversation context for a specific user or all users"""
        return self.orchestrator.reset_conversation_context(user_id)

    def upload_document(self, file_path, file_name, user_id=None, policy_id=None, document_type=None):
        """Upload and process a document"""
        # Check if file exists
        if not os.path.exists(file_path):
            return {"status": "error", "message": "File not found"}

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.config.MAX_DOCUMENT_SIZE:
            return {"status": "error", "message": f"File too large. Maximum size is {self.config.MAX_DOCUMENT_SIZE/(1024*1024)}MB"}

        # Check file extension
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in self.config.SUPPORTED_EXTENSIONS:
            return {"status": "error", "message": f"Unsupported file type. Supported types are: {', '.join(self.config.SUPPORTED_EXTENSIONS)}"}

        # Process the document
        result = self.document_processor.process_file(file_path, file_name, user_id, policy_id, document_type)

        if result["status"] == "success":
            # Update the knowledge base with key information from the document
            doc_knowledge = {
                "type": "policy_document",
                "policy_id": result.get("policy_number"),
                "document_type": result.get("document_type"),
                "file_name": file_name,
                "user_id": user_id,
                "description": f"Policy document {file_name} with ID {result.get('policy_number') or 'unknown'}"
            }
            self.add_knowledge(doc_knowledge)

            # Train the RAG system with the new document
            self.knowledge_base.compute_embeddings()
            self.knowledge_base.save_database()

        return result

    def get_user_documents(self, user_id):
        """Get all documents for a specific user"""
        return self.document_processor.get_documents_by_user(user_id)

    def get_document_by_id(self, doc_id):
        """Retrieve a document by ID"""
        return self.document_processor.get_document_by_id(doc_id)



# --- 10. Demo Usage ---

# Initialize the system
advisor = InsuranceAIAdvisor()

# Example queries
queries = [
    {
        "user_id": "user123",
        "query": "I'm 35, married with two kids ages 5 and 7. I make about $90k a year as a software developer. We own a house worth $450k with a $300k mortgage. What insurance should I have?"
    },
    {
        "user_id": "user123",
        "query": "Can you explain what a deductible is in simple terms?"
    },
    {
        "user_id": "user456",
        "query": "I'm starting a small consulting business from home. What insurance do I need?"
    },
    {
        "user_id": "user456",
        "query": "What's the difference between term life and whole life insurance?"
    }
]

# Run example queries
for example in queries:
    #print(f"\n\n--- QUERY: {example['query']} ---\n")
    response = advisor.query(example['user_id'], example['query'])
    #print(f"\n--- RESPONSE ---\n{response}")


# In[ ]:


# --- 11. Interactive Mode ---

"""def run_interactive_mode():
    user_id = "interactive_user"
    display(Markdown("Insurance AI Advisor - Interactive Mode"))
    display(Markdown("Type 'exit' to quit"))
    display(Markdown("Type 'new chat' to clear conversation history"))
    display(Markdown("Type 'reset' to reset the conversation context"))

    while True:
        user_input = input("\nYour query: ")
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() in ['new chat', 'clear chat', 'start over']:
            response = advisor.reset_context()
            print(f"\nResponse: {response}")
        elif user_input.lower() == 'reset':
            response = advisor.reset_context()
            print(f"\nResponse: {response}")
        else:

            response = advisor.query(user_id, user_input)
            display(Markdown(f"\nResponse: {response}"))

# For interactive notebook usage, uncomment:
run_interactive_mode()"""




# In[ ]:


# --- File Upload and Document Demo ---
def run_document_demo():
    """Demo for document upload and querying with documents"""
    from IPython.display import display, Markdown, FileLink, FileUpload
    import ipywidgets as widgets
    import tempfile
    import os
    import shutil

    # Initialize the system
    advisor = InsuranceAIAdvisor()
    user_id = "demo_user"

    # Create a temporary directory for uploads
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for uploads: {temp_dir}")

    # Display title
    display(Markdown("# Insurance AI Advisor with Document Processing"))
    display(Markdown("Upload insurance policy documents to enhance the advisor's knowledge"))

    # Create file upload widget
    upload_widget = FileUpload(
        description='Upload Documents',
        accept='.pdf,.docx,.jpg,.jpeg,.png,.txt',
        multiple=True
    )
    display(upload_widget)

    # Create a function to handle uploads
    def handle_upload(change):
        for name, file_info in upload_widget.value.items():
            print(f"Processing uploaded file: {name}")

            # Save the file to the temporary directory
            file_path = os.path.join(temp_dir, name)
            with open(file_path, 'wb') as f:
                f.write(file_info['content'])

            # Process the file
            result = advisor.upload_document(
                file_path=file_path,
                file_name=name,
                user_id=user_id
            )

            # Display the result
            if result['status'] == 'success':
                display(Markdown(f"✅ Successfully processed {name}"))
                display(Markdown(f"Detected Policy ID: {result.get('policy_number', 'Unknown')}"))
                display(Markdown(f"Document Type: {result.get('document_type', 'Unknown')}"))
            else:
                display(Markdown(f"❌ Error processing {name}: {result.get('error', 'Unknown error')}"))

    # Attach the handler to the upload widget
    upload_widget.observe(handle_upload, names='value')

    # Create a query input
    query_text = widgets.Text(
        value='',
        placeholder='Enter your insurance question here...',
        description='Query:',
        disabled=False,
        layout=widgets.Layout(width='80%')
    )

    query_button = widgets.Button(
        description='Ask',
        disabled=False,
        button_style='primary',
        tooltip='Ask your question',
        icon='question'
    )

    # Display the query input
    display(widgets.HBox([query_text, query_button]))

    # Output area for responses
    output_area = widgets.Output()
    display(output_area)

    # Handler for the query button
    def handle_query(b):
        query = query_text.value
        if not query:
            return

        with output_area:
            display(Markdown(f"**Your query:** {query}"))
            response = advisor.query(user_id, query)
            display(Markdown(f"**Response:** {response}"))
            print("\n" + "-"*50 + "\n")

    # Attach the handler to the button
    query_button.on_click(handle_query)

    # Display document list button
    doc_list_button = widgets.Button(
        description='List My Documents',
        disabled=False,
        button_style='info',
        tooltip='List all your uploaded documents',
        icon='list'
    )
    display(doc_list_button)

    # Handler for the document list button
    def handle_doc_list(b):
        with output_area:
            display(Markdown("## Your Documents"))
            docs = advisor.get_user_documents(user_id)
            if not docs:
                display(Markdown("No documents found. Please upload some documents."))
            else:
                for doc_id, metadata in docs.items():
                    display(Markdown(f"**Document:** {metadata.get('file_name', 'Unknown')}\n"
                                    f"**Type:** {metadata.get('document_type', 'Unknown')}\n"
                                    f"**Policy ID:** {metadata.get('policy_id', 'Unknown')}\n"
                                    f"**Upload Time:** {metadata.get('upload_time', 'Unknown')}\n"
                                    f"---"))

    # Attach the handler to the button
    doc_list_button.on_click(handle_doc_list)

    # Instructions
    instructions = """
    ## How to use:
    1. Upload your insurance policy documents using the upload button above
    2. Type your insurance-related questions in the query box
    3. Click 'Ask' to get personalized answers based on your uploaded documents
    4. Click 'List My Documents' to see all your uploaded documents

    Example queries:
    - "What are my health insurance coverage limits?"
    - "Does my auto policy cover rental cars?"
    - "What is my deductible for emergency medical under my travel insurance?"
    - "When does my current policy expire?"
    """
    display(Markdown(instructions))

    # Cleanup function
    def cleanup():
        """Clean up temporary files"""
        try:
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logging.info(f"Error cleaning up: {str(e)}")

    # Return the cleanup function
    return cleanup





