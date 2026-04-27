import os
from typing import Literal, List

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model.age_sex.wavlm_demographics import WavLMWrapper as WavLMWrapperAgeSex
from src.model.voice_quality.wavlm_voice_quality import WavLMWrapper as WavLMWrapperVoiceQuality


SEX_LABELS = [
    "female",
    "male",
]

VOICE_QUALITY_LABELS = [
    "shrill", "nasal", "deep",
    "silky", "husky", "raspy", "guttural", "vocal-fry",
    "booming", "authoritative", "loud", "hushed", "soft",
    "crisp", "slurred", "lisp", "stammering",
    "singsong", "pitchy", "flowing", "monotone", "staccato", "punctuated", "enunciated", "hesitant",
]


class PredictRequest(BaseModel):
    audio: List[float]
    sampling_rate: int = Field(default=16000, ge=1)
    return_value: Literal["logit", "prob", "logprob"] = "prob"


class PredictResponse(BaseModel):
    outputs: dict


def create_app() -> FastAPI:
    device = os.getenv("MODEL_DEVICE", "cpu")

    app = FastAPI(title="Demographics and Voice Quality API")

    age_sex_model = WavLMWrapperAgeSex.from_pretrained(
        "tiantiaf/wavlm-large-age-sex"
    ).eval().to(device)

    voice_quality_model = WavLMWrapperVoiceQuality.from_pretrained(
        "tiantiaf/wavlm-large-voice-quality"
    ).eval().to(device)

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        try:
            audio_np = np.asarray(req.audio, dtype=np.float32)
            if audio_np.ndim != 1:
                raise HTTPException(status_code=400, detail="Audio must be a 1D mono waveform list.")

            if req.sampling_rate != 16000:
                audio_np = librosa.resample(audio_np, orig_sr=req.sampling_rate, target_sr=16000)

            wave = torch.tensor(audio_np, dtype=torch.float32)[None].to(device)

            with torch.no_grad():
                age_pred, sex_pred = age_sex_model(wave)
                sex_prob = F.softmax(sex_pred, dim=1)
                sex_logprob = F.log_softmax(sex_pred, dim=1)

                age_pred = age_pred.squeeze().item()
                sex_pred_list = sex_pred.squeeze().tolist()
                sex_prob_list = sex_prob.squeeze().tolist()
                sex_logprob_list = sex_logprob.squeeze().tolist()

                vq_pred = voice_quality_model(wave, return_feature=False)
                vq_prob = torch.sigmoid(vq_pred)
                vq_logprob = torch.log(vq_prob)

                vq_pred_list = vq_pred.squeeze().tolist()
                vq_prob_list = vq_prob.squeeze().tolist()
                vq_logprob_list = vq_logprob.squeeze().tolist()

            if req.return_value == "logit":
                outputs = {
                    "age": age_pred,
                    **{label: sex_pred_list[i] for i, label in enumerate(SEX_LABELS)},
                    **{label: vq_pred_list[i] for i, label in enumerate(VOICE_QUALITY_LABELS)},
                }
            elif req.return_value == "prob":
                outputs = {
                    "age": age_pred,
                    **{label: sex_prob_list[i] for i, label in enumerate(SEX_LABELS)},
                    **{label: vq_prob_list[i] for i, label in enumerate(VOICE_QUALITY_LABELS)},
                }
            else:
                outputs = {
                    "age": age_pred,
                    **{label: sex_logprob_list[i] for i, label in enumerate(SEX_LABELS)},
                    **{label: vq_logprob_list[i] for i, label in enumerate(VOICE_QUALITY_LABELS)},
                }

            return PredictResponse(outputs=outputs)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


app = create_app()