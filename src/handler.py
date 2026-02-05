from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
import json, os, logging
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from utils import format_chat_prompt, create_error_response
from models import GenerationRequest, GenerationResponse, ChatCompletionRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine: Optional[AsyncLLMEngine] = None
engine_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_engine()
    yield
    global engine, engine_ready
    engine = None
    engine_ready = False

app = FastAPI(title="vLLM Qwen3-32B Load Balancer", lifespan=lifespan)

async def create_engine():
    global engine, engine_ready
    try:
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen-3-32B")
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            dtype="bf16",
            trust_remote_code=True,
            gpu_memory_utilization=0.95
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        engine_ready = True
        logger.info(f"Engine ready: {model_name}")
    except Exception as e:
        engine_ready = False
        logger.error(f"Engine initialization failed: {e}")
        raise

@app.get("/ping")
async def health_check():
    return JSONResponse(
        content={"status": "healthy" if engine_ready else "initializing"},
        status_code=200 if engine_ready else 204
    )

@app.post("/v1/completions", response_model=GenerationResponse)
async def generate_completion(request: GenerationRequest):
    if not engine_ready or engine is None:
        raise HTTPException(status_code=503, detail=create_error_response("ServiceUnavailable", "Engine not ready").model_dump())
    
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        stop=request.stop
    )

    request_id = random_uuid()
    try:
        if request.stream:
            return StreamingResponse(
                stream_completion(request.prompt, sampling_params, request_id),
                media_type="text/event-stream"
            )
        else:
            results = engine.generate(request.prompt, sampling_params, request_id)
            final_output = None
            async for output in results:
                final_output = output
            if final_output is None:
                raise HTTPException(status_code=500, detail=create_error_response("GenerationError", "No output generated", request_id).model_dump())
            o = final_output.outputs[0]
            return GenerationResponse(
                text=o.text,
                finish_reason=o.finish_reason,
                prompt_tokens=len(final_output.prompt_token_ids),
                completion_tokens=len(o.token_ids),
                total_tokens=len(final_output.prompt_token_ids)+len(o.token_ids)
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=create_error_response("GenerationError", str(e), request_id).model_dump())

async def stream_completion(prompt: str, sampling_params: SamplingParams, request_id: str):
    try:
        results = engine.generate(prompt, sampling_params, request_id)
        async for output in results:
            for o in output.outputs:
                yield f"data: {json.dumps({'text': o.text, 'finish_reason': o.finish_reason})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not engine_ready or engine is None:
        raise HTTPException(status_code=503, detail=create_error_response("ServiceUnavailable", "Engine not ready").model_dump())
    prompt = format_chat_prompt(request.messages, os.getenv("MODEL_NAME", "Qwen/Qwen-3-32B"))
    sampling_params = SamplingParams(max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p, stop=request.stop)
    request_id = random_uuid()
    try:
        results = engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for output in results:
            final_output = output
        o = final_output.outputs[0]
        return {
            "id": request_id,
            "object": "chat.completion",
            "model": os.getenv("MODEL_NAME", "Qwen/Qwen-3-32B"),
            "choices": [{"index":0, "message":{"role":"assistant","content":o.text}, "finish_reason": o.finish_reason}],
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(o.token_ids),
                "total_tokens": len(final_output.prompt_token_ids)+len(o.token_ids)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=create_error_response("ChatCompletionError", str(e), request_id).model_dump())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 80)), log_level="info")
