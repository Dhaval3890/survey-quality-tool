from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .validation import load_table_from_upload, basic_validation
from .reliability import reliability_report

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Survey quality tool is running!", "docs": "/docs"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    likert_min: float = Form(1.0),
    likert_max: float = Form(5.0),
    id_columns: Optional[str] = Form(""),
    straight_line_threshold: int = Form(6),
):
    """
    Upload survey data (CSV or Excel) and get:
    - basic validation
    - Cronbach's alpha (all numeric items)
    """
    raw = await file.read()
    df = load_table_from_upload(raw, file.filename)

    id_cols: List[str] = (
        [c.strip() for c in id_columns.split(",") if c.strip()]
        if id_columns
        else []
    )

    validation = basic_validation(
        df,
        likert_min=likert_min,
        likert_max=likert_max,
        id_cols=id_cols,
        straight_line_threshold=straight_line_threshold,
    )
    reliability = reliability_report(df)

    return JSONResponse(
        {
            "file": file.filename,
            "likert_min": likert_min,
            "likert_max": likert_max,
            "id_columns": id_cols,
            "validation": validation,
            "reliability": reliability,
        }
    )
