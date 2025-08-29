from ..versions.api.services.controllers.onnx_controller import ONNXService

def get_onnx_service() -> ONNXService:
    return ONNXService()