import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

client_llm = OpenAI(api_key=OPENAI_API_KEY)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY
)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "Advance-langchain"

try:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="page",
        field_schema="integer"
    )
except Exception:
    pass


def generate_multi_queries(user_query):
    system_prompt = """
You are an AI assistant that generates 3 different reformulations 
of a user query for multi-query RAG.

Rules:
1. Return only one question at a time.
2. Input will be "next question" after first query.
3. All questions must be different.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_query})

    question_list = []

    for _ in range(3):
        response = client_llm.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        generated_question = response.choices[0].message.content.strip()
        question_list.append(generated_question)
        messages.append({"role": "assistant", "content": "next question"})

    return question_list


def retrieve_pages(query):
    query_vector = embedder.embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[],
        query=query_vector,
        limit=3,
        with_payload=True
    )

    ranked_list = []

    for hit in results.points:
        page = hit.payload.get("page")
        if page is not None:
            ranked_list.append(page)

    return ranked_list


def calculate_rrf(rankings, k=60):
    rrf_scores = {}

    for rank_list in rankings:
        for position, doc_id in enumerate(rank_list):
            rank = position + 1
            score = 1 / (k + rank)

            if doc_id in rrf_scores:
                rrf_scores[doc_id] += score
            else:
                rrf_scores[doc_id] = score

    sorted_rrf = dict(
        sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_rrf


def fetch_page_content(page_number):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=None,
        limit=50,
        with_payload=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="page",
                    match=MatchValue(value=page_number)
                )
            ]
        )
    )

    page_text = ""
    for hit in results.points:
        page_text += hit.payload.get("text", "") + "\n"

    return page_text


if __name__ == "__main__":

    user_query = input("Enter your question: ")

    question_list = generate_multi_queries(user_query)

    all_rankings = []
    for q in question_list:
        ranked_pages = retrieve_pages(q)
        all_rankings.append(ranked_pages)

    final_ranking = calculate_rrf(all_rankings)

    print(final_ranking)
    context=""
    if final_ranking:
        if len(final_ranking.keys())>=2:
            for x in range(2):
                top_page = list(final_ranking.keys())[x]
                page_content = fetch_page_content(top_page)
                context+=page_content[:2000] + f"page number {top_page}  "
        else:
            top_page = list(final_ranking.keys())[0]
            page_content = fetch_page_content(top_page)
            context+=page_content[:2000] + f"page number {top_page}  "



    def final_prediction(context,user_query):
        system_prompts=f"""
        You are an Ai assistant, who answer the user query from the given context :

        context={context}
        """
        
        response=client_llm.chat.completions.create(
            model = "gpt-4o",
            messages=[{"role":"system","content":system_prompts},
                      {"role":"user","content":user_query}]
        )

        return response.choices[0].message.content
    

    res=final_prediction(context,user_query)
    print(res)