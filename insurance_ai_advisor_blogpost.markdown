# Transforming Insurance Choices with InsuranceAIAdvisor: A Gen AI-Powered Solution

Hey there, insurance explorers! Choosing the right health, auto, or travel insurance can feel like deciphering a labyrinth of jargon and fine print. It’s overwhelming, right? That’s why I created the **InsuranceAIAdvisor**, a Python-based Jupyter Notebook powered by Google’s Gemini model, leveraging Retrieval-Augmented Generation (RAG), ReAct reasoning, and agent-based architecture. With a sleek Streamlit interface, it delivers personalized advice and policy analysis that’s easy to understand. Let’s dive into how it simplifies insurance, explore the code driving it, and peek at future implementations. Buckle up—this is a game-changer!

## The Problem: Insurance Decisions Are a Puzzle

Navigating insurance is no easy feat. Customers wrestle with complex terms like “deductible” or “coinsurance,” unsure which plan suits their needs—perhaps a robust health plan for a family or travel coverage for a globetrotter. This challenge is especially pronounced for those seeking guidance on insurance terms, conditions, and the specific coverages they should have, such as understanding policy limits or identifying gaps in existing plans. Brokers and insurers, meanwhile, need tools to provide quick, tailored recommendations to help clarify these details. The `InsuranceAIAdvisor` tackles this with generative AI, offering a smart solution that processes queries, analyzes uploaded policies, and suggests customized coverage for health, auto, and travel insurance, making it ideal for users looking to demystify insurance specifics.

## How Gen AI Makes It Simple

The `InsuranceAIAdvisor` is like your personal insurance guide, blending five generative AI capabilities across its components:

- **Config**: Centralizes settings like the Google API key, file paths (`knowledge_base/`, `user_profiles/`, `documents/`), and rules (20MB file size limit, supported formats: PDF, DOCX, TXT, JPG, PNG, JPEG). It sets agent parameters like `MAX_STEPS=5` for ReAct reasoning and `VERBOSE` for detailed logs.
- **setup_models**: Initializes the Gemini-1.5-flash model for text generation and `GoogleEmbeddingModel` for embeddings, using the Config’s API key.
- **GoogleEmbeddingModel**: Generates 768-dimensional embeddings for texts, with batch processing (size 5) and rate limiting (0.5s delay), falling back to zero vectors on errors.
- **KnowledgeBase**: Stores insurance data (terms, policies, FAQs) in `contents.json` and embeddings in `embeddings.npy`, enabling cosine similarity searches for top_k=3 results.
- **DocumentProcessor**: Extracts policy details (e.g., policy numbers, coverage limits) from PDFs (PyPDF2), DOCX (docx), or images (pytesseract OCR), chunking text for analysis.
- **UserProfileManager**: Extracts structured profiles (age, medical history, vehicle type) from text using Gemini or regex, storing them in `profiles.json`.
- **RAGSystem**: Combines `KnowledgeBase` and `DocumentProcessor` searches to augment LLM prompts, filtering results by 0.5 similarity for relevance.
- **InsuranceAgent**: Drives ReAct reasoning (up to 5 steps), supporting actions like knowledge search, profile analysis, or option comparison.
- **RecommendationAgent & ExplainerAgent**: Specialized agents for suggesting policies (e.g., high-coverage auto plans for risky drivers) or explaining terms with analogies and examples.
- **AgentOrchestrator**: Coordinates queries with a single API call, caching responses and tracking context to handle topic shifts (e.g., health to travel).
- **InsuranceAIAdvisor**: The main interface, integrating all components for queries, document uploads, and profile updates.

Users can ask, “Does my policy cover rental cars?” or upload a PDF policy, and the system delivers answers grounded in their data, enhanced by a Streamlit UI with pages for querying, uploading documents, viewing policies, and managing profiles.

## The Streamlit UI: Your Gateway to Insurance Clarity

The Streamlit interface makes `InsuranceAIAdvisor` a joy to use. It offers five pages—Home, Query AI, Upload Document, View Documents, and User Profile—accessible via a sidebar with a user ID input (default: `guest_user`). Custom CSS styles the app with a clean look (light gray background, rounded buttons). The **Home** page welcomes users, outlining features like document analysis and personalized advice. **Query AI** accepts questions, displays responses with conversation history, and includes a reset option. **Upload Document** supports multi-file uploads, validates formats, and processes files. **View Documents** lists user documents with metadata (e.g., policy ID, coverage limits). **User Profile** offers a form to update details like age or medical history. Session state tracks interactions, and robust error handling ensures a smooth experience.

### Where to Add UI Screenshots

To showcase the Streamlit UI, include screenshots of key pages (e.g., Query AI, Upload Document, View Documents). Here’s how:

1. **Capture Images**: Take screenshots of the Streamlit app (e.g., Home page, Query AI with a response, Upload Document success message, User Profile form).
2. **Host Images**: Upload to a free host like Imgur or a GitHub repository. Copy direct image URLs (ending in `.png` or `.jpg`).
3. **Embed in Gist**: Add markdown image syntax in your Gist at these points:
   - After “The Streamlit UI” section: `![Streamlit Home Page](https://i.imgur.com/your_image_link.png)`
   - After describing Query AI: `![Query AI Page](https://i.imgur.com/your_image_link.png)`
   - After Upload Document: `![Upload Document Page](https://i.imgur.com/your_image_link.png)`
   - After View Documents: `![View Documents Page](https://i.imgur.com/your_image_link.png)`
4. **Gist Limitation**: GitHub Gist doesn’t host images, so external hosting is required. Link to a GitHub repo folder with images in the Gist description for reference.

## Code Snippets: Inside the Engine

Let’s explore the code powering this system.

### Snippet 1: Setting Up Models

```python
def setup_models(config):
    import google.generativeai as genai
    genai.configure(api_key=config.GOOGLE_API_KEY)
    llm = genai.GenerativeModel('gemini-1.5-flash')
    embedding_model = GoogleEmbeddingModel(config.GOOGLE_API_KEY)
    return llm, embedding_model
```

This initializes the Gemini model and embedding model for text generation and vector searches.

### Snippet 2: Document Processing

```python
def process_file(self, file_path, file_name, user_id=None):
    ext = os.path.splitext(file_name)[1].lower()
    if ext == '.pdf':
        text_content = self._extract_pdf_text(file_path)
    policy_number = self._extract_policy_number(text_content)
    doc_id = f"doc_{int(time.time())}_{file_name}"
    self.metadata[doc_id] = {"policy_id": policy_number, "user_id": user_id}
    self.save_metadata()
    return {"doc_id": doc_id, "status": "success"}
```

This processes uploaded files, extracting policy details for analysis.

### Snippet 3: RAG Query

```python
def query(self, user_query, user_profile=None, user_id=None, top_k=3):
    expanded_query = user_query
    if user_profile:
        profile_summary = f"User is {user_profile.get('age', '')} years old"
        expanded_query = f"{profile_summary}. Query: {user_query}"
    kb_results, kb_scores = self.knowledge_base.search(expanded_query, top_k=top_k)
    context = ""
    for result, score in zip(kb_results, kb_scores):
        if score < 0.5:
            continue
        if result["type"] == "policy":
            context += f"POLICY: {result['name']}\nDESCRIPTION: {result['description']}\n"
    return context
```

This fetches relevant context from the knowledge base for precise answers.

### Snippet 4: ReAct Reasoning

```python
def react(self, user_query, user_profile=None):
    prompt = self._create_react_prompt(user_query, user_profile)
    thoughts = []
    for step in range(self.config.MAX_STEPS):
        response = self.llm.generate_content(prompt)
        thought, action, action_input = self._parse_react_response(response.text)
        thoughts.append({"thought": thought, "action": action, "action_input": action_input})
        if action == "Final Answer":
            return self._ground_response(action_input)
        action_result = self._perform_action(action, action_input)
        prompt += f"\nObservation: {action_result}\nThought:"
    return thoughts[-1]["thought"]
```

This implements ReAct reasoning for complex queries, iterating up to 5 steps.

### Snippet 5: Profile Extraction

```python
def extract_profile_from_text(self, text):
    prompt = f"""
    Extract structured profile data from this text in JSON format:
    {text}
    Fields: age, occupation, income, family_status, medical_history, vehicle_type
    """
    response = self.llm.generate_content(prompt)
    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
    return json.loads(json_match.group(0)) if json_match else {}
```

This extracts structured user data for personalization.

## Future Implementations

The `InsuranceAIAdvisor` is powerful but has room to grow. It relies on Gemini’s API, which can hit quota limits under heavy use, slowing responses. Regex-based profile extraction in `UserProfileManager` may miss nuanced details (e.g., complex medical conditions). The `KnowledgeBase` requires manual updates, risking outdated data. The `DocumentProcessor` is currently in a testing/beta stage, with OCR accuracy varying for low-quality images, requiring further refinement for production readiness. Product selection—allowing users to choose specific insurance products from a catalog—is not yet implemented, limiting direct policy purchasing.

Exciting future enhancements include:

- **Multi-model support**: Integrate open-source LLMs to reduce API dependency.
- **Dynamic knowledge updates**: Use web scraping to keep the `KnowledgeBase` current.
- **Real-time pricing APIs**: Add cost estimates for recommendations.
- **Enhanced OCR for DocumentProcessor**: Leverage advanced vision models to improve image processing, moving it out of beta.
- **Product selection interface**: Implement a catalog for users to browse and select insurance products directly.
- **Multilingual UI**: Support non-English users with translated interfaces.
- **Broker dashboard**: Visualize customer trends in the Streamlit app for brokers.

## Why It’s a Game-Changer

The `InsuranceAIAdvisor` is innovative, blending document understanding, ReAct agents, and a user-friendly Streamlit UI to simplify a complex process. It’s particularly suited for users seeking clarity on insurance terms, conditions, and coverages, making it ideal for insurance companies as a chatbot, where customers can upload policies for tailored comparisons, or for brokers’ websites, generating insights (e.g., health plans for seniors) to streamline consultations. The five Gen AI capabilities—RAG, document understanding, agents, embeddings, and structured output—ensure accuracy and scalability, making it a practical, impactful tool.

## Try It Out!

The notebook is on GitHub—fire it up, upload a policy, or ask about “coinsurance.” The Streamlit UI makes it a joy to use. Fork the repo, share your thoughts, or add a feature! What’s your take on simplifying insurance with AI?

**Word Count**: 1015