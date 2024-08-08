from fastapi import FastAPI
from inference_realesrgan_aws import main
from pydantic import BaseModel


class data(BaseModel):
	img_str: str
	half: bool = False
	output_dir: str = "results"



app = FastAPI()

@app.get("/")
def read_root():
	return {"Hello": "World"}



@app.post("/inference")
def inference(item: data):
	result = main(item.img_str)
	
	return result

