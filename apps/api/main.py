# apps/api/main.py
from apps.api.app_factory import create_app
from apps.api.jobs import router as jobs_router

app = create_app()
app.include_router(jobs_router)
