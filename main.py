from fastapi import FastAPI, Depends
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


# 모델과 토크나이저를 로드하는 의존성(dependency) 함수를 정의합니다.
def get_model_tokenizer():
    model, tokenizer = model_settings()
    return model, tokenizer

# 의존성을 라우트 함수에 사용합니다.
@app.post("/get_prediction/")
async def get_prediction(input: str, model_tokenizer=Depends(get_model_tokenizer)):
    model, tokenizer = model_tokenizer
    output = gen(model, tokenizer, input)
    return {'output': output}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=40001, reload=True)