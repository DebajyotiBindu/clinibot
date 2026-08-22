# CliniBot
### Your virtual Assistant for helping you out with labels and dosages of various clinical drugs of your choice.

## 1. Problem Statement
In clinical settings, patient leaflets (Product Labels) are dense, legalistic, and difficult for the average patient to parse. Misinterpretation of dosage instructions, contraindications (e.g., aspirin sensitivity), or off-label use poses significant health risks. Existing AI tools often "hallucinate" medical advice, making them unreliable for clinical contexts.

## 2. Our Solution
**CliniBot** is a Retrieval-Augmented Generation (RAG) system designed to provide **grounded, context-aware, and safe responses** to medical label queries. By enforcing a strict retrieval pipeline from verified clinical PDFs, we ensure every response is cited, verifiable, and strictly limited to the provided document’s scope.

## 3. Tech Stack & Architecture
| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | LangChain | Manages the RAG pipeline and stateful chat memory. |
| **LLM Engine** | Groq (Qwen/Qwen3.6-27b) | High-throughput inference for rapid, low-latency responses. |
| **Data Ingestion** | PyPDF | Extracts raw text while maintaining document structure. |
| **Embeddings** | HuggingFace | Transforms clinical text into high-dimensional semantic vectors. |
| **Vector DB** | ChromaDB | Efficient semantic indexing and similarity retrieval. |
| **Evaluation** | Pydantic / LLM-as-a-Judge | Provides programmatic, structured validation of AI safety. |

## 4. Architectural Workflow
<img width="1027" height="631" alt="Screenshot 2026-08-21 164310" src="https://github.com/user-attachments/assets/1eb5f43f-5c43-4109-97c3-c52c3d1e491b" />

## 5. Folder Structure
<img width="352" height="440" alt="image" src="https://github.com/user-attachments/assets/9c298d35-ff89-4e92-bb91-b67689d91c3c" />

## 5. Project Scripts Description
*   **`src/loading.py`**: Handles raw text extraction from PDF labels, ensuring document structure and metadata (like page numbers) are preserved for citations.
*   **`src/embedding.py`**: Converts clinical text into semantic vector representations using state-of-the-art embedding models, optimized for medical terminology.
*   **`src/retrieval.py`**: The orchestration layer. It manages the semantic search against the clinical vector database and retrieves context-relevant chunks.
*   **`evaluation.py`**: The evaluating layer. It measures and compares the performance of various instance of models under different parameters and setups.
*   **`quality_check.py`**: Our automated evaluation suite. It executes "Golden Set" tests, forces structured JSON validation, and generates performance reports.

## 6. Evaluation Framework
We validate system performance using an **LLM-as-a-Judge** framework based on three clinical benchmarks:
*   **Context Precision:** Did we retrieve only relevant clinical sections?
*   **Context Recall:** Did we retrieve all information necessary to answer the prompt?
*   **Groundedness:** Does the model's answer stay strictly within the provided document, or is it hallucinating?
*   **Query Latency:** Is the model fast enough to respond to the user's query within 3 seconds time frame limit?

### The Golden Set
Our **Golden Set** is a curated list of question-answer pairs that test the "Four Pillars of Clinical RAG":
1.  **Dosing:** Accuracy of numerical limits (e.g., 1200mg/24h).
2.  **Contraindications:** Safety-critical logic (e.g., aspirin allergy).
3.  **Indications:** Scoping (What the drug actually treats).
4.  **Negative Constraints:** Adversarial handling (e.g., "Can I use this for a broken arm?").

### App Interface
<img width="1401" height="686" alt="Screenshot 2026-08-20 130637" src="https://github.com/user-attachments/assets/94921a4e-772f-4eec-b54d-7494bb79df18" />

### Evaluation results
<img width="648" height="320" alt="Screenshot 2026-08-20 191541" src="https://github.com/user-attachments/assets/78ea7262-f6bd-4791-b72c-53702ecb1df9" />
<img width="1031" height="315" alt="Screenshot 2026-08-20 191554" src="https://github.com/user-attachments/assets/c5aefb2f-cd37-45a6-bd17-dd1bda849cd4" />
<img width="1054" height="332" alt="Screenshot 2026-08-20 191703" src="https://github.com/user-attachments/assets/8ab36606-3cce-41f7-8060-167985706475" />
<img width="1042" height="324" alt="Screenshot 2026-08-20 191714" src="https://github.com/user-attachments/assets/baf138af-fc3b-47f9-a5bb-4f9596b22ed0" />
<img width="1002" height="306" alt="Screenshot 2026-08-20 191728" src="https://github.com/user-attachments/assets/37203b47-6fd5-4d10-9b4b-d11cf375a5fc" />
<img width="1089" height="386" alt="Screenshot 2026-08-20 194954" src="https://github.com/user-attachments/assets/9cf8fb76-d9f9-463b-90c0-0709c40686e1" />
<img width="1090" height="307" alt="Screenshot 2026-08-22 090728" src="https://github.com/user-attachments/assets/85e229cc-9b96-4d51-8109-dee0b30de003" />


## 7. Getting Started
### Prerequisites
*   Python 3.10+
*   Groq API Key
*   `pip install -r requirements.txt`

### Running the Project
1.  **Environment Setup**: Create a `.env` file and add your `GROQ_API_KEY`.
2.  **Ingestion**: Ensure your PDF label is in the source directory.
3.  **Run CliniBot**:
    ```bash
    streamlit run app.py
    ```
4. **Run Evaluation**:
    ```bash
    python quality_check.py
    ```
4.  **View Results**: The script generates a `manual_evaluation_report.csv` file, providing a detailed audit trail of system accuracy and groundedness.

## 7. Safety Disclaimer
*This tool is designed for educational purposes and provides information based solely on the supplied documents. It does not replace the professional advice of a physician. Always consult a healthcare provider for medical emergencies or diagnosis.*
