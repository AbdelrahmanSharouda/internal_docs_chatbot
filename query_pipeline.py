from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load your FAISS vector index
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# 2. Initialize Llama model
llm = LlamaCpp(
    model_path="models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    # verbose=True
)

# 3. Define a prompt template
SYSTEM_TEMPLATE = """
Answer the user's question based on the context below.
If the context doesn't contain relevant information, just say "I don't know".

<context>
{context}
</context>
"""

prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Build the RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Ask a question!
question = input("Ask me anything: ")
response = rag_chain.invoke(question)
print("🤖 Answer:", response)
