from pydantic import BaseModel
from typing import List, Any

class ONNXRequest(BaseModel):
    messages: List[Any]
    # query: str

class ONNXResponse(BaseModel):
    content: str
    prompt_tokens: int
    generated_tokens: int