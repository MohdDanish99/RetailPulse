# Vercel entrypoint: exposes the FastAPI app at the project root
from src.api.main import app  # noqa: F401
