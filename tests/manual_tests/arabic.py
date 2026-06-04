from mtn_agents.upload_reference import upload_file_to_index_pinecone, upload_url_to_index_pinecone
from mtn_agents.retriever import PineconeIndex
from mtn_agents.chains import ChainV2

user_id = "arabic_manual_test"
path_to_arabic_doc = "<>"
url_to_arabic_website = "<>"
doc1 = "arabic_doc"
doc2 = "arabic_url"

data_file = upload_file_to_index_pinecone(local_path=path_to_arabic_doc, user_id=user_id, document_id=doc1)
data_url = upload_url_to_index_pinecone(url_address=url_to_arabic_website, user_id=user_id, document_id=doc2)

question = ""
rag = ChainV2()
pc = PineconeIndex()

answer = rag.qa(chat_history="", question=question, user_id=user_id, document_ids=[doc1, doc2])

pc.delete_user_id(user_id=user_id)
