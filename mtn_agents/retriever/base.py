from pydantic import BaseModel


class BaseIndex(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def upsert(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
