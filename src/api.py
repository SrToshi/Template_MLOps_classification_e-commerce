"""
api.py — HTTP contract for the Rakuten project (Phase 1).

This file exposes two endpoints, backed by the same logic already built
and verified in training.py and predict.py:

    POST /training/   -> triggers a full retraining run
    POST /predict/     -> returns predictions for a batch of products

FastAPI (the framework) is just the transport layer here. The actual
ML logic lives, unchanged, in training.py / predict.py.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import json

from tensorflow import keras

from predict import Predict
from training import run_training


app = FastAPI(title="Rakuten Classification API")


# ---------------------------------------------------------------------------
# WHY THIS SECTION EXISTS (read this before touching /predict/)
#
# predict.py's `main()` function loads the tokenizer, the LSTM model and
# the VGG16 model EVERY time it runs — that's fine for a one-off CLI call,
# but wrong for an API: if we repeated that inside the /predict/ endpoint,
# EVERY single HTTP request would reload a full VGG16 model from disk
# (several seconds, as we measured yesterday) before doing any actual
# prediction. With many requests, that cost repeats every time for no
# reason — the model weights never change between requests.
#
# The fix: load everything ONCE, here, at import time (i.e. when the
# server starts up, before it accepts any request). Each individual
# request then only does the actual inference step, reusing these
# already-loaded objects. This is why the code below looks similar to
# predict.py's main(), but lives at module level instead of inside a
# per-request function.
# ---------------------------------------------------------------------------

with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
TOKENIZER = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

LSTM_MODEL = keras.models.load_model("models/best_lstm_model.h5")
VGG16_MODEL = keras.models.load_model("models/best_vgg16_model.h5")

with open("models/best_weights.json", "r") as json_file:
    BEST_WEIGHTS = json.load(json_file)

with open("models/mapper.json", "r") as json_file:
    MAPPER = json.load(json_file)

# --- end of the "load once" block ------------------------------------------


class PredictRequest(BaseModel):
    """What a client must send to POST /predict/."""
    dataset_path: str = "data/preprocessed/X_train_update.csv"
    images_path: str = "data/preprocessed/image_train"


@app.post("/predict/")
def predict_endpoint(request: PredictRequest):
    # Note the contrast with the block above: here we build a NEW `Predict`
    # instance per request, but we pass in the models we already loaded at
    # startup (TOKENIZER, LSTM_MODEL, VGG16_MODEL, ...) instead of loading
    # them again. Only what's genuinely specific to this one request
    # (which CSV, which images folder) is created fresh each time.
    predictor = Predict(
        tokenizer=TOKENIZER,
        lstm=LSTM_MODEL,
        vgg16=VGG16_MODEL,
        best_weights=BEST_WEIGHTS,
        mapper=MAPPER,
        filepath=request.dataset_path,
        imagepath=request.images_path,
    )
    predictions = predictor.predict()
    return {"predictions": predictions}


@app.post("/training/")
def training_endpoint():
    # Unlike /predict/, this one is intentionally simple: run_training()
    # already IS the whole pipeline (load data, train LSTM, train VGG16,
    # optimize the ensemble, save every artifact to models/). There is
    # nothing to "load once" here, because a training run is not something
    # you want to reuse across requests — every call is meant to produce a
    # fresh model.
    #
    # Known limitation for Phase 1 (fine for now, worth revisiting in
    # Phase 2): this call blocks the request for as long as training
    # takes (~2h45min on your machine yesterday). A production version
    # would kick this off as a background job (e.g. FastAPI
    # BackgroundTasks, or a proper task queue) and return immediately
    # with a "started" status instead of making the caller wait.
    result = run_training()
    return result