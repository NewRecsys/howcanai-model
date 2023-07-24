from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from polyglot import model_settings, gen

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, tokenizer = model_settings()

@app.post("/get_prediction/")
async def get_prediction(input: str):
    output = gen(model, tokenizer, input)
    return {'output': output}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=30013, reload=True)