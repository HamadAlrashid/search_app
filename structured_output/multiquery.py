from pydantic import BaseModel

class MultiQuery(BaseModel):
    queries: list[str]

    def __str__(self):
        return ",".join(self.queries)

    def __iter__(self):
        return iter(self.queries)
    
    def __len__(self):
        return len(self.queries)

    
    
    