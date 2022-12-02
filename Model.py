from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
t = pipeline("translation_ru_to_en", model="facebook/wmt19-ru-en")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/translate/")
def predict(item: Item):
    return t(item.text)[0]