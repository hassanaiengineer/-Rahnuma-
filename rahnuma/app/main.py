from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .model import qalb
from .graph import workflow
import uvicorn
import os

app = FastAPI(title="Rahnuma — اردو AI اسسٹنٹ")

# Enable CORS for local UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import List, Optional

class ChatRequest(BaseModel):
    user_message: str
    history: Optional[List[str]] = []

@app.on_event("startup")
async def startup_event():
    qalb.load()

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.user_message:
        raise HTTPException(status_code=400, detail="پیغام خالی نہیں ہو سکتا")
        
    try:
        inputs = {
            "user_input": request.user_message,
            "history": request.history or [],
            "safety_passed": True
        }
        result = workflow.invoke(inputs)
        return {"assistant_reply": result["response"]}
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return {"assistant_reply": "کچھ مسئلہ پیش آیا، دوبارہ کوشش کریں"}

# Serve UI
ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
