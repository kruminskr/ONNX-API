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
        response = await onnx_service.get_gpt2_onnx(request.query)

        return response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error occurred while fetching data")