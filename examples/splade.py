import os

import dspy
from dotenv import load_dotenv

load_dotenv(override=True)

lm = dspy.LM(
    "openai/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    api_base="https://api.mistral.ai/v1/",
)
dspy.configure(lm=lm)


def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url="http://0.0.0.0:8000/search/single")(query, k=3)
    return [x["text"] for x in results]


rag = dspy.ChainOfThought("context, question -> response")

question = "What's the name of the castle that David Gregory inherited?"
ans = rag(context=search_wikipedia(question), question=question)
print(ans)
