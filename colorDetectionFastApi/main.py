from uvicorn import Server, Config
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os

from routers import mediapipe

# app = FastAPI()
app = FastAPI(docs_url="/swagger")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mediapipe.router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server = Server(Config(app, host="0.0.0.0", port=port, lifespan="on"))
    server.run()