from agents import planner, retriever, reranker, generator

def main():
    print("🧠 Agentic RAG - Anxiety Chatbot")
    query = input("❓ Posez votre question : ").strip()
    if not query:
        print("⚠️ Question vide.")
        return

    plan = planner.plan_query(query)
    print(f"🧭 Plan détecté : {plan}")

    retrieved_docs = retriever.retrieve(query, strategy=plan["retrieval_strategy"])
    if not retrieved_docs:
        print("❌ Aucun document trouvé.")
        return

    reranked_docs = reranker.rerank(query, retrieved_docs)
    response = generator.generate(query, reranked_docs)

    print("🤖 Réponse générée :")
    print(response)

if __name__ == "__main__":
    main()