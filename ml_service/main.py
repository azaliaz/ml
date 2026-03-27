from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)
import os

import uvicorn

from .api import app


if __name__ == "__main__":
    module_app = (
        "ml_service.main:app"
        if Path(__file__).resolve().parent.name == "ml_service"
        else "main:app"
    )
    uvicorn.run(
        module_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
