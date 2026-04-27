# %%
import torch
import torch.nn.functional as F

import librosa

from src.model.age_sex.wavlm_demographics import WavLMWrapper as WavLMWrapperAgeSex
from src.model.voice_quality.wavlm_voice_quality import WavLMWrapper as WavLMWrapperVoiceQuality

# %%
# Label list
SEX_LABELS = [
    'female',
    'male',
]
VOICE_QUALITY_LABELS = [
    'shrill', 'nasal', 'deep',  # Pitch
    'silky', 'husky', 'raspy', 'guttural', 'vocal-fry', # Texture
    'booming', 'authoritative', 'loud', 'hushed', 'soft', # Volume
    'crisp', 'slurred', 'lisp', 'stammering', # Clarity
    'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated', 'hesitant', # Rhythm
]

# %%
# Load models
device = "cpu"

age_sex_model = WavLMWrapperAgeSex.from_pretrained("tiantiaf/wavlm-large-age-sex").eval().to(device)
voice_quality_model = WavLMWrapperVoiceQuality.from_pretrained("tiantiaf/wavlm-large-voice-quality").eval().to(device)

# %%
# Input data would be audio torch.tensor, mono channel, shape 1 x n_samples
# I'm loading a wav file just for example. In the API, it must accept a list of floats.
wave_path = "/workspace/decoding/assets/misc/LJ037-0171.wav"
wave, _ = librosa.load(wave_path, sr=16_000)
wave = torch.tensor(wave)[None]

wave.shape

# %%
return_value = "prob" # logit, prob, logprob

with torch.no_grad():
    # Predict age and sex
    age_pred, sex_pred = age_sex_model(wave)
    sex_prob = F.softmax(sex_pred, dim=1)
    sex_logprob = F.log_softmax(sex_pred, dim=1)
    age_pred, sex_pred = age_pred.squeeze().item(), sex_pred.squeeze().tolist()
    sex_prob, sex_logprob = sex_prob.squeeze().tolist(), sex_logprob.squeeze().tolist()
    # Predict voice quality
    vq_pred = voice_quality_model(wave, return_feature=False)
    vq_prob = F.sigmoid(vq_pred)
    vq_logprob = vq_prob.log()
    vq_pred, vq_prob, vq_logprob = vq_pred.squeeze().tolist(), vq_prob.squeeze().tolist(), vq_logprob.squeeze().tolist()

if return_value == "logit":
    outputs = {
        **{"age": age_pred},
        **{l: sex_pred[i] for i, l in enumerate(SEX_LABELS)},
        **{l: vq_pred[i] for i, l in enumerate(VOICE_QUALITY_LABELS)},
    }

elif return_value == "prob":
    outputs = {
        **{"age": age_pred},
        **{l: sex_prob[i] for i, l in enumerate(SEX_LABELS)},
        **{l: vq_prob[i] for i, l in enumerate(VOICE_QUALITY_LABELS)},
    }

elif return_value == "logprob":
    outputs = {
        **{"age": age_pred},
        **{l: sex_logprob[i] for i, l in enumerate(SEX_LABELS)},
        **{l: vq_logprob[i] for i, l in enumerate(VOICE_QUALITY_LABELS)},
    }

outputs

# %%


# %%


# %%



