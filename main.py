from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routes.user import router

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "templates" / "static"

print("BASE_DIR:", BASE_DIR)
print("STATIC_DIR:", STATIC_DIR)
print("STATIC_DIR exists:", STATIC_DIR.exists())
print("user.png exists:", (STATIC_DIR / "user.png").exists())
print("chatbot.png exists:", (STATIC_DIR / "chatbot.png").exists())

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

app.include_router(router)