from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from app.validation import validate_survey
from app.reliability import compute_reliability
from app.json_safe import to_json_safe  # 🔥 make results JSON-safe
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Survey Quality Tool API")


@app.get("/")
def root():
    return {"message": "Survey quality tool is running!"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    likert_min: int = Form(...),
    likert_max: int = Form(...),
    id_columns: str = Form(...),
    straight_line_threshold: int = Form(...)
):

    logger.info(f"Analyzing file: {file.filename}")
    contents = await file.read()
    df = pd.read_csv(
        pd.io.common.BytesIO(contents),
        encoding="ISO-8859-1"
    )

    id_cols = [col.strip() for col in id_columns.split(",")]

    # Run quality checks
    validation = validate_survey(df, id_cols, likert_min, likert_max, straight_line_threshold)
    reliability = compute_reliability(df, id_cols)

    result = {
        "file": file.filename,
        "total_rows": len(df),
        "likert_min": likert_min,
        "likert_max": likert_max,
        
        # Results
        "validation": validation,
        "reliability": reliability,
    }

    # 🔥 Fix error:
    # ValueError: Out of range float values are not JSON compliant
    safe_result = to_json_safe(result)

    return JSONResponse(content=safe_result)
