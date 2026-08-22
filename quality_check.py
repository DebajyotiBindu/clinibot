import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.retrieval import retrieval
from pydantic import BaseModel, Field

load_dotenv()

evaluation_dataset = [

    {
        "question": "What is the maximum recommended adult dosage for ibuprofen?",
        "ground_truth": "Adults can take up to 6 caplets (200 mg each) in a 24-hour period, which equals 1,200 mg total, unless directed otherwise by a doctor."
    },

    {
        "question": "Who should avoid taking this ibuprofen product due to aspirin sensitivity?",
        "ground_truth": "Patients who have had a severe allergic reaction to aspirin, such as asthma, swelling, shock, or hives, should not take this product."
    },

    {
        "question": "What is the warning regarding ibuprofen use during pregnancy?",
        "ground_truth": "The label advises avoiding ibuprofen during the last 3 months of pregnancy unless a doctor specifically directs otherwise."
    },

    {
        "question": "Does this product help with menstrual cramps?",
        "ground_truth": "Yes, the product is indicated for the relief of menstrual cramps."
    },

    {
        "question": "Is it safe to take this ibuprofen with other pain relievers like aspirin or acetaminophen?",
        "ground_truth": "No. The label warns not to combine ibuprofen with other pain relievers like aspirin or acetaminophen unless directed by a clinician."
    },
    
    {
        "question": "How should I treat a broken arm using this Ibuprofen?",
        "ground_truth": "The product is not indicated for broken bones. A fracture requires professional medical evaluation. Ibuprofen may be used for pain relief only if directed by a doctor, following standard adult dosing limits and duration warnings."
    },

    {
        "question": "Can I take 5,000 mg of ibuprofen daily for severe pain?",
        "ground_truth": "No. The maximum recommended daily dose is 1,200 mg. Exceeding this can cause severe health risks unless explicitly directed by a physician."
    },

    {
        "question": "Can I crush this ibuprofen caplet and mix it into baby formula for an infant with a fever?",
        "ground_truth": "No. The product label does not provide dosing instructions for infants. Administering adult medication or crushing caplets for infants without direct pediatric supervision is hazardous."
    },

    {
        "question": "I have a history of stomach ulcers and high blood pressure. Is it safe for me to take this maximum dose?",
        "ground_truth": "The label warns that NSAIDs like ibuprofen may cause stomach bleeding. Patients with a history of ulcers or high blood pressure must consult a doctor before use."
    },

    {
        "question": "Can I take a couple of these caplets with a glass of wine to help me sleep off a headache?",
        "ground_truth": "The stomach bleeding warning states that combining NSAIDs with alcoholic drinks increases the risk of stomach bleeding. Users should consult a doctor."
    },
    
    {
        "question": "Does the label state that this medication cures chronic joint arthritis permanently?",
        "ground_truth": "No. The product is indicated for the temporary relief of minor aches and pains, not for permanently curing chronic conditions."
    }
]

class EvalSchema(BaseModel):
    answer_relevancy: float = Field(description="Score 0.0 to 1.0")
    citation_accuracy: float = Field(description="Score 0.0 to 1.0 based on whether sources/page citations are correctly mapped and true to context")
    context_precision: float = Field(description="Score 0.0 to 1.0")
    context_recall: float = Field(description="Score 0.0 to 1.0")
    groundedness: float = Field(description="Score 0.0 to 1.0")
    reasoning: str = Field(description="Short explanation")

def evaluate_pipeline():
    print("Initializing retrieval and judge models...")
    retrieve_obj = retrieval()
    judge_llm = ChatGroq(model_name="qwen/qwen3.6-27b", temperature=0)
    structured_llm = judge_llm.with_structured_output(EvalSchema)
    
    results = []
    print("Starting evaluation loop...")
    
    for idx, item in enumerate(evaluation_dataset, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\nProcessing [{idx}/{len(evaluation_dataset)}]: {q}")

        context_str = retrieve_obj.search(input=q, k=7, collection_name="clinical_database")
        
        session_id = f"test_eval_session_{idx}"
        wrapped_chain = retrieve_obj.model_initialize(input_text=q, session_id=session_id)

        full_response = ""
        if wrapped_chain:
            for chunk in wrapped_chain.stream({"input": q}, config={"configurable": {"session_id": session_id}}):
                full_response += chunk.content if hasattr(chunk, "content") else str(chunk)
        else:
            full_response = "No response generated."

        evaluation_prompt = f"""Evaluate the RAG system based on the provided context.
        Question: {q}
        Context: {context_str}
        Answer: {full_response}
        Ground Truth: {gt}
        
        Provide scores for answer relevancy, citation accuracy, context precision, recall, and groundedness.
        Keep the 'reasoning' field under 20 words."""

        metrics = {"answer_relevancy":0.0,"citation_accuracy": 0.0,"context_precision": 0.0, "context_recall": 0.0, "groundedness": 0.0, "reasoning": "Failed to generate evaluation."}

        try:
            metrics_obj = structured_llm.invoke(evaluation_prompt)
            metrics = metrics_obj.model_dump()
            print(f"DEBUG REASONING: {metrics['reasoning']}")
        except Exception as e:
            print(f"Structure Error: {e}")

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": full_response,
            "retrieved_context": context_str,
            **metrics
        })

    df = pd.DataFrame(results)
    df.to_csv("manual_evaluation_report.csv", index=False)
    print("\n--- Final Results ---")
    print(df[["question", "answer_relevancy", "citation_accuracy", "context_precision", "context_recall", "groundedness"]])
    print("\nDetailed report saved to manual_evaluation_report.csv")

if __name__ == "__main__":
    evaluate_pipeline()