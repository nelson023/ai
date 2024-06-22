#使用CHATGPT編寫程式碼 看懂了並無修改
from langchain.llms import OpenAI
from langchain.chains import RAGChain
from langchain.schema import Prompt
from langchain.retrievers import ElasticsearchRetriever

# 初始化 OpenAI 模型
llm = OpenAI()

# 設定 Elasticsearch 檢索器
retriever = ElasticsearchRetriever(
    index_name="your_index_name",  # 請替換成您的索引名稱
    es_host="http://localhost:9200",  # Elasticsearch 服務地址
    es_user="your_username",  # 使用者名稱，如果有的話
    es_password="your_password"  # 密碼，如果有的話
)

# 創建 RAG 鏈
rag_chain = RAGChain(llm=llm, retriever=retriever)

# 使用 RAG 鏈生成答案
prompt = Prompt("What is the capital of France?")
response = rag_chain.run(prompt)

print("Response:", response.output)
