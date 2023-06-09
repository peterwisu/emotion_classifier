import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from model import EmoClassifier
import typing
import logging
import os


# request object
class Request(BaseModel):

    text:  str

# returning object
class Result(BaseModel):

    text: str
    label: str
    score: float

# response object
class Response(BaseModel):

    results: typing.List[Result] 


logging.basicConfig(filename="log_file.log",
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s ,%(levelname)s:, user_input: %(user_input)s, model_prediction: %(model_prediction)s, score: %(score)s',
                    
                    )



model_type = os.environ.get("MY_MODEL")
# initialise predictor
emo_predictor = EmoClassifier(logger=logging,model_type=model_type)

# # create UI
demo = gr.Interface(
        fn=emo_predictor.predict_emo,
        inputs=gr.Textbox(label="Ask a question"),
        outputs=[gr.Textbox(label="text"),gr.Textbox(label="Answer"),gr.Number(label="Score")],
        allow_flagging="never",
        )



# initialise server 
app = FastAPI()


# API endpoint for prediction
@app.post("/predict", response_model=Response)
async def predict_api(request: Request):
    
    print(request.text)
    results =  emo_predictor.predict_emo(request.text)

    return Response(
                       results=[ Result(text=results[0],label=results[1],score=results[2])]
                    )

# # mount UI on the server
app = gr.mount_gradio_app(app, demo, path="/")


