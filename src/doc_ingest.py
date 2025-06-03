import os
import re
import shutil
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
        


def get_env_variable(var_name):
    value = os.environ.get(var_name)
    if not value:
        raise EnvironmentError(f"Environment variable {var_name} is not set.")
    return value


def clean_text(text):
    # remove URL 
    text = re.sub(r'https?://\S+', '', text)
    # remove page number i.e. 1/10
    text = re.sub(r'^\s*\d+\s*\/\d+$', '', text, flags=re.MULTILINE)
    # remove time stamps i.e. 12/31/2023, 10:30 AM
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(AM|PM)?', '', text)
    # remove verbose copyright notices
    text = re.sub(r'Carro Malaysia Terms of Use', '', text, flags=re.IGNORECASE)
    # remove excessive empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def is_level1_title(line):
    # Short line, 4-60 characters, no punctuation, at least 2 words with 60% capitalized
    line = line.strip()
    if len(line) < 4 or len(line) > 60:
        return False
    if re.search(r"[,.:;?!]", line):
        return False
    words = line.split()
    if len(words) >= 2:
        cap_count = sum([w[0].isupper() for w in words if w[0].isalpha()])
        if cap_count >= max(1, int(0.6 * len(words))) and words[0][0].isupper():
            return True
    return False

def is_level2_title(line):
    # "PART A :", "PART B1 :"" ... in a single line
    return bool(re.match(r"^PART\s+[A-Z0-9]+:$", line.strip()))

def is_level3_title(line):
    # "1. " or "a. " in a single line
    return bool(re.match(r"^\d+\.\s", line.strip()) or re.match(r"^[a-zA-Z]\.\s", line.strip()))

def is_toc_start(line):
    return "comprises of" in line.lower() or "table of contents" in line.lower()

def is_part_line(line):
    return bool(re.match(r"^\d+\.\s*Part\s+[A-Z0-9]+[ :]?", line.strip()))


def chunk_pdf_text_to_docs(text, min_chunk_len=60, min_merge_len=120):
    lines = text.split('\n')
    level1, level2, level3 = "", "", ""
    chunk_buffer = []
    docs = []
    toc_mode = False
    toc_buffer = []

    # title chunk without contents
    pending_title = {"level1": None, "level2": None, "level3": None}

    def flush_chunk():
        nonlocal chunk_buffer, level1, level2, level3, pending_title
        chunk_text = " ".join(chunk_buffer).strip()
        if len(chunk_text) >= min_chunk_len:
            meta = {
                "level1_title": pending_title["level1"] or level1,
                "level2_title": pending_title["level2"] or level2,
                "level3_title": pending_title["level3"] or level3,
            }
            docs.append(Document(page_content=chunk_text, metadata=meta))
            pending_title = {"level1": None, "level2": None, "level3": None}
        chunk_buffer = []

    for line in lines:
        line = line.strip()

        if is_toc_start(line):
            toc_mode = True
            toc_buffer = [line]
            continue
        if toc_mode:
            if is_part_line(line):
                toc_buffer.append(line)
                continue
            else:
                # end of toc mode - generate toc chunk
                if toc_buffer:
                    docs.append(Document(
                        page_content="\n".join(toc_buffer),
                        metadata={"type": "toc", "level1_title": level1}
                    ))
                    toc_buffer = []
                toc_mode = False
                # no return here, continue processing the next line


        if is_level1_title(line):
            flush_chunk()
            if chunk_buffer:
                level1, level2, level3 = line, "", ""
            else:
                pending_title["level1"] = line
                level1, level2, level3 = line, "", ""
        elif is_level2_title(line):
            flush_chunk()
            if chunk_buffer:
                level2, level3 = line, ""
            else:
                pending_title["level2"] = line
                level2, level3 = line, ""
        elif is_level3_title(line):
            if len(" ".join(chunk_buffer).strip()) >= min_merge_len:
                flush_chunk()
            if chunk_buffer:
                level3 = line
            else:
                pending_title["level3"] = line
                level3 = line
        else:
            if line:
                chunk_buffer.append(line)
    
    flush_chunk()
    if toc_mode and toc_buffer:
        docs.append(Document(
            page_content="\n".join(toc_buffer),
            metadata={"type": "toc", "level1_title": level1}
        ))

    # merge short chunk
    merged_docs = []
    buffer = None
    for doc in docs:
        if buffer is None:
            buffer = doc
        elif len(buffer.page_content) < min_merge_len or len(doc.page_content) < min_merge_len:
            buffer = Document(
                page_content=buffer.page_content + " " + doc.page_content,
                metadata=buffer.metadata
            )
        else:
            merged_docs.append(buffer)
            buffer = doc
    if buffer:
        merged_docs.append(buffer)
    return merged_docs

def read_pdf_file(file_path):
    all_docs = []
    for filename in os.listdir(file_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(file_path, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if not docs:
                continue
            
            union_doc = docs[0] if docs else None
            union_doc.metadata["filetype"] = 'pdf'
            union_doc.metadata["source"] = filename
            union_doc.page_content = "\n".join([doc.page_content for doc in docs])  # 合并内容
            # print(union_doc.page_content[:3000])
            
            clean_text_content = clean_text(union_doc.page_content)
            # print(clean_text_content[:3000])
            
            struc_docs = chunk_pdf_text_to_docs(clean_text_content)
            # for i, doc in enumerate(struc_docs[:10], 1):
            #     title_parts = [doc.metadata.get('level1_title'), doc.metadata.get('level2_title'), doc.metadata.get('level3_title')]
            #     title = " | ".join([x for x in title_parts if x not in (None, "")])
            #     content = f"[{title}]\n" + doc.page_content.replace('\n', ' ')
            #     print(f"Chunk {i}")
            #     print(f"  Level1: {doc.metadata.get('level1_title')}")
            #     print(f"  Level2: {doc.metadata.get('level2_title')}")
            #     print(f"  Level3: {doc.metadata.get('level3_title')}")
            #     print(f"  Content ({len(doc.page_content)}): {content}...")
            #     print("-" * 80)
                
            all_docs.extend(struc_docs)

    # print(f"Loaded {len(all_docs)} documents.")
    return all_docs


def sliding_chunk_with_metadata_delimiter(all_docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".", "!", "?", "\n", ":"]
    )
    all_chunks = []
    for doc in all_docs:
        # title prefix
        meta = dict(doc.metadata)
        titles = [meta.get(k) for k in ["level1_title", "level2_title", "level3_title"] if meta.get(k)]
        prefix = f"[{' | '.join(titles)}]\n" if titles else ""
        # split section
        for split in text_splitter.split_text(doc.page_content):
            chunk_text = prefix + split
            all_chunks.append(Document(page_content=chunk_text, metadata=meta))
    print(f"Split into {len(all_chunks)} chunks.")
    return all_chunks


def simple_chunk(all_docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".", "!", "?", "\n", ":"]
    )
    split_docs = splitter.split_documents(all_docs)
    # print(f"Split into {len(split_docs)} chunks.")
    return split_docs


def faiss_embed(all_docs, index_dir):
    """
    Creates a FAISS index from the documents and saves it to the INDEX_DIR.
    """
    split_docs = sliding_chunk_with_metadata_delimiter(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(index_dir)
    
    print(f"FAISS index saved to: {index_dir}")
    print("Total Vectors: ", vectorstore.index.ntotal)
    print("Total Dimensions: ", vectorstore.index.d)
    print("Index Type: ", type(vectorstore.index))


def ingest_documents(data_dir, index_dir):
    """
    Ingests PDF documents from the DATA_DIR, splits them into chunks, and creates a FAISS index.
    """
    all_docs = read_pdf_file(data_dir)
    if not all_docs:
        raise ValueError("No documents found to ingest.")

    # shutil.rmtree(index_dir, ignore_errors=True)
    faiss_embed(all_docs, index_dir)
    
    # from doc_retrieve import list_chunks
    # list_chunks(index_dir)


