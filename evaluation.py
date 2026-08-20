import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith import evaluate
from langsmith.schemas import Run, Example
from langchain_groq import ChatGroq
from src.retrieval import retrieval
import json

client = Client()
dataset_name = "clinical-rag-eval-dataset-v2"

if not client.has_dataset(dataset_name=dataset_name):
    print(f"Creating dataset '{dataset_name}' on LangSmith...")
    client.create_dataset(
        dataset_name=dataset_name, 
        description="Comprehensive clinical RAG test cases covering dosages, warnings, and contraindications."
    )
    client.create_examples(
        dataset_name=dataset_name,
        examples=[
            {
                "inputs": {"question": "What is the maximum recommended adult dosage for ibuprofen?"},
                "outputs": {"ground_truth": "The maximum amount of ibuprofen for adults is 800 milligrams per dose or 3200 mg per day unless directed otherwise by a doctor."}
            },
            {
                "inputs": {"question": "What are the warning signs or contraindications for aspirin-sensitive patients regarding NSAIDs?"},
                "outputs": {"ground_truth": "Patients who have had an asthma attack, hives, or severe allergic reaction after taking aspirin or other NSAIDs should avoid ibuprofen."}
            },
            {
                "inputs": {"question": "What are the warnings regarding heart surgery and NSAIDs like ibuprofen?"},
                "outputs": {"ground_truth": "Ibuprofen should not be used right before or after coronary artery bypass graft (CABG) heart surgery because it increases the risk of heart attack or stroke."}
            },
            {
                "inputs": {"question": "What specific precautions should pregnant women take with NSAIDs like ibuprofen?"},
                "outputs": {"ground_truth": "Do not take ibuprofen at 20 weeks or later in pregnancy unless directed by a doctor, as it can cause serious heart or kidney problems in the unborn baby."}
            },
            {
                "inputs": {"question": "What symptoms of stomach bleeding should prompt a patient to stop taking ibuprofen and seek medical help?"},
                "outputs": {"ground_truth": "Stop taking the medication and seek help if you feel faint, vomit blood, have bloody or black stools, or experience persistent stomach pain."}
            },
            {
                "inputs": {"question": "What is the maximum recommended daily dose of acetaminophen for healthy adults?"},
                "outputs": {"ground_truth": "Healthy adults should not exceed 4,000 mg of acetaminophen in a 24-hour period to avoid severe liver damage."}
            },
            {
                "inputs": {"question": "Who should consider lowering their daily limit of acetaminophen?"},
                "outputs": {"ground_truth": "Individuals with pre-existing liver disease, heavy alcohol use, malnutrition, or older adults should limit intake to 2,000 to 3,000 mg daily."}
            },
            {
                "inputs": {"question": "What is 'double-dipping' when taking OTC pain or cold medications?"},
                "outputs": {"ground_truth": "Taking multiple products simultaneously that both contain hidden acetaminophen, which can cause an accidental overdose."}
            },
            {
                "inputs": {"question": "What emergency symptoms of an allergic reaction require immediate medical attention?"},
                "outputs": {"ground_truth": "Hives, facial swelling, difficulty breathing, wheezing, or a severe skin rash with blistering."}
            },
            {
                "inputs": {"question": "What are the primary cardiovascular risks associated with long-term NSAID use?"},
                "outputs": {"ground_truth": "An increased risk of serious cardiovascular thrombotic events, including fatal heart attack and stroke."}
            }
        ]
    )

retrieve_obj = retrieval()
judge_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

def target_pipeline(inputs: dict) -> dict:
    q = inputs["question"]
    session_id = "langsmith_eval_session"

    context_str = retrieve_obj.search(input=q, k=5, collection_name="clinical_database")
    wrapped_chain = retrieve_obj.model_initialize(input_text=q, session_id=session_id)

    full_response = ""
    if wrapped_chain:
        for chunk in wrapped_chain.stream(
            {"input": q},
            config={"configurable": {"session_id": session_id}}
        ):
            if hasattr(chunk, "content"):
                full_response += chunk.content
            else:
                full_response += str(chunk)
    else:
                full_response="No response generated from the model."

    return {
        "answer": full_response,
        "contexts": [context_str] if context_str else [""]
    }

def llm_semantic_correctness(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("answer", "")
    ground_truth = example.outputs.get("ground_truth", "")
    question = example.inputs.get("question", "")
    
    prompt = f"""You are an objective clinical evaluation judge. Compare the AI's generated answer against the ground truth reference for the given question.
    
    Question: {question}
    Ground Truth Reference: {ground_truth}
    AI Generated Answer: {prediction}
    
    Task: Determine if the AI answer is factually aligned and semantically correct based on the ground truth. 
    Output ONLY a JSON format with two keys: "score" (either 1.0 if correct/acceptable or 0.0 if incorrect/hallucinated) and "reason" (a short explanation).
    """
    
    try:
        response = judge_llm.invoke(prompt)
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        result_json = json.loads(content)
        score = float(result_json.get("score", 0.0))
        reason = result_json.get("reason", "Evaluated by LLM judge.")
    except Exception as e:
        score = 0.5
        reason = f"Parsing error during evaluation: {str(e)}"

    return {
        "key": "semantic_correctness",
        "score": score,
        "comment": reason
    }

if __name__ == "__main__":
    print(f"Running LangSmith evaluation experiment with LLM Judge on dataset: {dataset_name}...")
    
    experiment_results = evaluate(
        target_pipeline,
        data=dataset_name,
        evaluators=[llm_semantic_correctness],
        experiment_prefix="clinical-rag-llm-judge",
        description="Evaluating clinical RAG pipeline using an LLM-as-a-judge semantic evaluator."
    )

    
