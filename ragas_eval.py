import time
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from app import build_qa_pipeline

EVAL_CSV = "eval_questions.csv"


def invoke_with_retry(fn, *args, retries=6, base_wait=25):
    """
    Retries when Gemini throws 429 (RESOURCE_EXHAUSTED).
    Keeps it simple: waits and tries again.
    """
    for attempt in range(retries):
        try:
            return fn(*args)
        except ChatGoogleGenerativeAIError as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = base_wait + attempt * 5
                print(f"[429 quota] Waiting {wait}s then retrying... (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            raise  # other errors should crash
    raise RuntimeError("Failed after retries due to repeated quota limits.")


def main():
    qa, retriever = build_qa_pipeline()

    # keep it small at first, then increase
    df = pd.read_csv(EVAL_CSV, engine="python").head(5)

    questions, answers, contexts, gts = [], [], [], []

    for i, row in df.iterrows():
        q = str(row["question"]).strip()
        gt = str(row["ground_truth"]).strip()

        # retrieve contexts (usually no LLM call)
        docs = retriever.invoke(q)
        ctx = [d.page_content for d in docs]

        # generate answer (LLM call)
        a = invoke_with_retry(qa.invoke, q)

        questions.append(q)
        answers.append(a)
        contexts.append(ctx)
        gts.append(gt)

        # small pacing so you don't burst requests
        time.sleep(2)

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": gts
    })

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    # SUPER IMPORTANT: no parallel calls
    run_config = RunConfig(max_workers=1)

    result = evaluate(ds, metrics=metrics, run_config=run_config)

    print(result)
    print("\nScores:\n", result.to_pandas())


if __name__ == "__main__":
    main()
