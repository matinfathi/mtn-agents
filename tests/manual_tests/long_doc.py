from mtn_agents.upload_reference import upload_file_to_index_pinecone
from mtn_agents.retriever import PineconeIndex

pc = PineconeIndex()
long_document_path = "<Path to the long document>"
user_id = "long_doc_test"

data = upload_file_to_index_pinecone(local_path=long_document_path, user_id=user_id, document_id="long_doc")

pc.delete_user_id(user_id=user_id)
