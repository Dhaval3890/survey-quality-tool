from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .validation import load_table_from_upload, basic_validation, reliability_analysis
from .json_safe import to_json_safe

app = FastAPI()

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    likert_min: int = Form(...),
    likert_max: int = Form(...),
):
    try:
        df = load_table_from_upload(file)

        validation = basic_validation(df, likert_min, likert_max)
        reliability = reliability_analysis(df)

        result = {
            "validation": validation,
            "reliability": reliability
        }

        return JSONResponse(content=to_json_safe(result))

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
