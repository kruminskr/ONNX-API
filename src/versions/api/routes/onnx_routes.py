import traceback

from fastapi import APIRouter, Depends, HTTPException

from ....core.dependencies import get_onnx_service
from ..services.controllers.onnx_controller import ONNXService
from ..services.schemas.onnx_schemas import (
    ONNXRequest,
    ONNXResponse,
)

router = APIRouter()

@router.post("/gpt2", response_model=ONNXResponse)
async def process_gpt2_onnx(request: ONNXRequest, onnx_service: ONNXService = Depends(get_onnx_service)):
    try:
        prompt = request.messages[0]['content']

        response = onnx_service.generate(prompt, 100, 0.6, 30)

        return response
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"An error occurred: {e}")
        print(traceback_str)      

        raise HTTPException(status_code=500, detail="Error occurred while fetching data")