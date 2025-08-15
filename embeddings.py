from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.schema import Document
from pathlib import Path
import pandas as pd

def load_python_scripts(path):
    docs = []
    for filepath in Path(path).rglob("*.py"):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            docs.append(Document(page_content=content, metadata={"source": str(filepath)}))
    return docs

def load_sheets(sheet):
    # Load Tasks sheet
    docs = []
    df_tasks = pd.read_excel(sheet)
    for _, row in df_tasks.iterrows():
        text = "\n".join(f"{col}: {row[col]}" for col in df_tasks.columns)
        docs.append(Document(page_content=text, metadata={"source": "airflow_tasks_sheet"}))

    return docs

def load_docs(path):
    loader = UnstructuredFileLoader(path)
    return loader.load()

def load_dbt_files(path):
    docs = []
    for filepath in Path(path).rglob("*"):
        if filepath.suffix in [".sql", ".yml", ".yaml"]:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(Document(page_content=content, metadata={"source": str(filepath)}))
    return docs


dbt_docs = load_dbt_files("data/dbt_repo")
airflow_code_docs = load_python_scripts("data/airflow_enriched/DAGs")
airflow_sheet_docs_dags = load_sheets("data/Airflow-DAGs-Docs.xlsx")
airflow_sheet_docs_tasks = load_sheets("data/airflow_processing_jobs.xlsx")
snowflake_docs = load_docs("data/Snowflake Documentation.docx")
transfer_docs = load_docs("data/Data Transfer Docs.docx")
ebook_docs = load_docs("data/MaxAB Data Engineering E-book.docx")


all_docs = airflow_code_docs + airflow_sheet_docs_dags+ airflow_sheet_docs_tasks + snowflake_docs + dbt_docs + transfer_docs + ebook_docs
print(f"✅ Loaded {len(all_docs)} raw documents.")

# 2. Split it into small chunks
print("🔍 Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Embed and store
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)

# Save to disk (optional)
vectorstore.save_local("faiss_index")
print("✅ Embeddings stored in FAISS!")
