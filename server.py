from fastapi import FastAPI, HTTPException
from texual_visual_merger import merger
app = FastAPI()

@app.post("/summerize")
def predict(data: dict):
    try:
        input_path = data["input_path"]
        output_path = data["output_path"]
        if not merger(input_path, output_path):
            raise HTTPException(status_code=500, detail="AI Model Failed")

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="AI Model Failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9004)