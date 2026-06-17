import os
from unittest import result
from urllib import request
from torch import nn, optim
import torchaudio
from torchaudio.models.decoder import ctc_decoder
import numpy as np
import uuid
from datetime import datetime

from data import get_infer_data_loader
from models.model.early_exit import Early_conformer, full_conformer, Early_zipformer, Splitformer
from util.beam_infer import BeamInference
from util.conf import get_args
from util.data_loader import text_transform
from util.epoch_timer import epoch_time
from util.model_utils import *
from util.tokenizer import *
import torchaudio.transforms as T
import torch.nn.functional as F
import json

from huggingface_hub import snapshot_download

import nuclio_sdk

#############################################
# PARAMETRI AUDIO E FINESTRE
#############################################

SAMPLE_RATE = 16000


LOOKBEHIND_SEC = 2.5 #0.5
CHUNK_SEC      = 0.51 #2.0
LOOKAHEAD_SEC  = 0.5 #0.5

LB = int(LOOKBEHIND_SEC * SAMPLE_RATE)
CK = int(CHUNK_SEC      * SAMPLE_RATE)
LA = int(LOOKAHEAD_SEC  * SAMPLE_RATE)

TOTAL_LEN = LB + CK + LA

WINDOW = LB + CK + LA   # 4 secondi = 64000 campioni
ADVANCE = CK            # avanza di 2 secondi = 32000 campioni

# buffer PCM per accumulare voce
#pcm_buffer = deque(maxlen=SAMPLE_RATE * 60)  # 60 s max

# per segmentazione basata su heartbeat
SEGMENT_TIMEOUT = 2000 # 800 ms di soli heartbeat = fine segmento
FRAME_MS = 30
FRAME_TIMEOUT = SEGMENT_TIMEOUT / FRAME_MS

# parametri encoder
SUBSAMPLING = 4
FEAT_FPS = 100  # feature frame per secondo prima del subsampling
EMB_PS = FEAT_FPS // SUBSAMPLING
LB_e = int(LOOKBEHIND_SEC * EMB_PS )
CK_e = int(CHUNK_SEC  * EMB_PS)


def spec_transform(waveform, args):
    spec_t = T.Spectrogram(n_fft=args.n_fft * 2, hop_length=args.hop_length, win_length=args.win_length)
    return spec_t(waveform)


def melspec_transform(waveform, args):
    melspec_t = T.MelScale(sample_rate=args.sample_rate, n_mels=args.n_mels, n_stft=args.n_fft+1)
    return melspec_t(waveform)


def handler(context:nuclio_sdk.Context, event: nuclio_sdk.Event):
    context.logger.info(f"start request handler at {datetime.now().isoformat()}")
    audio_bytes = event.body
    model = getattr(context, 'model', None)
    args = getattr(context, 'args', None)
    inf = getattr(context, 'inf', None)
    #context.logger.info(f"Model: {type(model)}, Inference Utils: {inf}, Args: {args}")
    #context.logger.info(f"Current working directory: {os.getcwd()}")
    buffer = getattr(context, 'buffer', None)
    if buffer is None:
        buffer = np.zeros(0, dtype=np.float32)
        setattr(context, 'buffer', buffer)
    first_block = getattr(context, 'first_block', True)
    
    try:
    # create buffer window and update buffer
        if len(audio_bytes) > 0:
            pcm_buffer = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, pcm_buffer])
            window, buffer = build_window_from_buffer(buffer, final_flush=False)   
            setattr(context, 'buffer', buffer)
            #window = pcm_buffer
            #first_block = True
            if window is None: 
                return void_response()
            
            # --- Estrai SOLO la parte centrale (2 secondi) ---
            #central = window[LB : LB + CK]   # float32 [-1,1]
            wav = torch.from_numpy(window).unsqueeze(0)  # (1, T)
            spec = spec_transform(wav, args)
            spec = melspec_transform(spec, args).to(args.device)
            valid_len = torch.tensor([spec.size(2)])
            encoder = model(spec, valid_len)
            enc = encoder[5]   
            B, T_full, D = enc.shape
            enc_central = None
            if first_block:
                context.logger.info(f"first block")
                enc_central = enc
                setattr(context, 'first_block', False)
            else:
                enc_central = enc[:, LB_e : LB_e + CK_e, :]
            # decodifica
            transc = None
            transc = inf.stream_decoder(emission=enc_central, partial=True)
            # if dev == "cpu":
            #     transc = inf.ctc_predict_(encoder[5])
            # if dev == "cuda":        
            #     best_combined = inf.ctc_cuda_predict(encoder[5], args.tokens)
            #     transc = args.sp.decode(best_combined[0].tokens).lower()
            # Normalizza l’output
            if isinstance(transc, list):
                caption = " ".join(transc)      # <-- qui mettiamo gli spazi
            else:
                caption = str(transc)

            #transc = run(args, model, inf, audio_bytes)
            #caption = transc[0]
            context.logger.info(f"caption: {caption}")

            return context.Response(
                body=caption,
                headers={},
                content_type="application/json",
                status_code=200
            ) 
        else:
            context.logger.info("No audio data received in the request")
            return void_response()       
    except Exception as e:
        context.logger.error(f"Error processing audio: {e}")
        return context.Response(
            body=json.dumps({
                "error": str(e),
                "status": "error"
            }),
            content_type="application/json",
            status_code=500
        ) 


def void_response():
    caption = "..."
    return nuclio_sdk.Response(
        body=json.dumps({
            "outputs": [
                {
                    "name": "caption",
                    "datatype": "BYTES",
                    "shape": [1, len(caption)],
                    "data": [caption]
                }
            ]
        }),
        headers={},
        content_type="application/json",
        status_code=200
    )

def init_model(context:nuclio_sdk.Context, lang:str, device:str = "cpu"):
    #
    #   CONFIG
    #
    context.logger.info(f"Loading model for {lang}")
    context.logger.info(f"Current working directory: {os.getcwd()}")

    if not os.path.exists("model-conformer"):
        os.makedirs("model-conformer")

    match lang:
        case "en":
            snapshot_download(repo_id="SpeechTek/English-EE-conformer", local_dir="model-conformer")
        case "it":
            snapshot_download(repo_id="SpeechTek/Italian-EE-conformer", local_dir="model-conformer")
        case _:
            raise ValueError(f"Lingua non supportata: {lang}")
    context.logger.info("Model from HF downloaded")

    args = get_args()
    args.load_model_path = args.load_model_dir + "/model"
    
    # If model checkpoint path is provided, load it.
    # (Overrides conf parameters)
    
    args.batch_size=1
    args.device=device
    
    # Parse config from command line arguments

    # Define model
    print(args)

    model = Early_conformer(src_pad_idx=args.src_pad_idx,
                                    n_enc_exits=args.n_enc_exits,
                                    d_model=args.d_model,
                                    enc_voc_size=args.enc_voc_size,
                                    dec_voc_size=args.dec_voc_size,
                                    max_len=args.max_len,
                                    d_feed_forward=args.d_feed_forward,
                                    n_head=args.n_heads,
                                    n_enc_layers=args.n_enc_layers_per_exit,
                                    features_length=args.n_mels,
                                    drop_prob=args.drop_prob,
                                    depthwise_kernel_size=args.depthwise_kernel_size,
                                    device=args.device).to(args.device)
    context.logger.info("Conformer done")

    model_path=args.load_model_dir+"/model"
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    context.logger.info(f'The model has {count_parameters(model):,} trainable parameters')
    #torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.n_threads)
    
    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=args)
    context.logger.info("BeamInference done")

    # add model to context
    #run(model=model, args=args, inf=inf)
    setattr(context, "model", model)
    setattr(context, "args", args)
    setattr(context, "inf", inf)
    

def build_window_from_buffer(buffer, final_flush=False):
    """
    buffer: numpy array float32 mono
    LB_SAMPLES: look-behind in samples
    CK_SAMPLES: central chunk in samples
    LA_SAMPLES: look-ahead in samples
    final_flush: True quando il VAD dice che ha finito lo speech
    """

    LB_SAMPLES = LB
    CK_SAMPLES = CK
    LA_SAMPLES = LA
    total_needed = LB_SAMPLES + CK_SAMPLES + LA_SAMPLES

    if final_flush:

        if len(buffer) < CK_SAMPLES:
            return None, buffer
        else:
            LB_ = buffer[:LB_SAMPLES]
            CK_ = buffer[LB_SAMPLES : LB_SAMPLES + CK_SAMPLES]
            LA_ = buffer[LB_SAMPLES + CK_SAMPLES : len(buffer)]    
        
            window = np.concatenate([LB_, CK_, LA_])
            #print("W_FLUSH:",len(window))
            # Avanza il buffer di CK 
            buffer = buffer[CK_SAMPLES:]
            return window, buffer
    else:
        # Se non abbiamo abbastanza aspettiamo ancora look-ahead
        if len(buffer) < total_needed:
            return None, buffer

        # Caso normale: finestra completa
        LB_ = buffer[:LB_SAMPLES]
        CK_ = buffer[LB_SAMPLES : LB_SAMPLES + CK_SAMPLES]
        LA_ = buffer[LB_SAMPLES + CK_SAMPLES : ,]

        window = np.concatenate([LB_, CK_, LA_])
        
        # Avanza il buffer di CK 
        buffer = buffer[CK_SAMPLES:]

        return window, buffer
