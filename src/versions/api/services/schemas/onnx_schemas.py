from pydantic import BaseModel

class ONNXRequest(BaseModel):
    query: str

class ONNXResponse(BaseModel):
    message: str