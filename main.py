import os
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI  
from langchain_community.document_loaders import TextLoader

# Khởi tạo mô hình OpenAI với biến môi trường
openai_api_key = os.getenv("OPENAI_API_KEY")

# Dữ liệu mẫu để chatbot tìm kiếm thông tin
documents = [
    Document(page_content="Python là một ngôn ngữ lập trình phổ biến, dễ học và mạnh mẽ."),
    Document(page_content="Machine learning là lĩnh vực AI sử dụng thuật toán để học từ dữ liệu."),
    Document(page_content="FAISS là một thư viện mã nguồn mở của Facebook để tìm kiếm vector hiệu quả."),
    Document(page_content="LangChain là một framework hỗ trợ xây dựng ứng dụng AI dựa trên mô hình ngôn ngữ lớn.")
]

# Chia nhỏ văn bản để tối ưu việc tìm kiếm
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
split_docs = text_splitter.split_documents(documents)

# Tạo bộ nhớ vector FAISS với OpenAI Embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_docs, embeddings)

# Tạo hệ thống truy xuất tài liệu
retriever = vector_store.as_retriever()

# Khởi tạo chatbot sử dụng GPT-4 với cơ chế RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    chain_type="stuff",
    retriever=retriever
)

def chatbot():
    print("Chatbot RAG - Nhập 'exit' để thoát")
    while True:
        query = input("Bạn: ")
        if query.lower() == "exit":
            break
        response = qa_chain.invoke({"query": query})  
        print("Bot:", response["result"])  


if __name__ == "__main__":
    chatbot()
