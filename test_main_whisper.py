import logging
import nuclio_sdk
import whisper
import numpy as np
from datetime import datetime
from util.audio_utils import read_audio_to_lpcm_bytes

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s - %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def init_model(context, lang:str, device:str = "cpu"):
    #
    #   CONFIG
    #
    context.logger.info(f"Loading model for {lang}")
    model = whisper.load_model("tiny", device=device)
    setattr(context, 'model', model)


def handler(context:nuclio_sdk.Context, event: nuclio_sdk.Event):
    context.logger.info(f"start request handler at {datetime.now().isoformat()}")
    pcm_bytes = event.body
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    model = getattr(context, 'model', None)
    if model is not None:
        result = model.transcribe(audio)
        context.logger.info(f"transcription: {result['text']}")
        return result


def process_audio_in_chunks(context, lpcm_bytes, chunk_size, sample_rate=16000):
    """
    Divide l'array di byte LPCM in chunk e invoca il handler per ogni chunk.
    
    Args:
        context: Contesto nuclio
        lpcm_bytes (bytes): Array di byte LPCM
        chunk_size (int): Numero di byte per chunk
        sample_rate (int): Frequenza di campionamento (per logging)
    
    Esempio:
        >>> process_audio_in_chunks(context, lpcm_bytes, chunk_size=2048, sample_rate=16000)
    """
    num_chunks = (len(lpcm_bytes) + chunk_size - 1) // chunk_size
    logger.info(f"Processing {len(lpcm_bytes)} bytes in {num_chunks} chunks of {chunk_size} bytes each")
    
    results = []
    for i in range(0, len(lpcm_bytes), chunk_size):
        chunk = lpcm_bytes[i:i+chunk_size]
        chunk_num = (i // chunk_size) + 1
        logger.info(f"Processing chunk {chunk_num}/{num_chunks} - Size: {len(chunk)} bytes")
        
        event = nuclio_sdk.Event(body=chunk)
        try:
            result = handler(context, event)
            results.append(result)
            logger.info(f"Chunk {chunk_num} processed successfully")
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
            results.append(None)
    
    logger.info(f"All {num_chunks} chunks processed")
    return results

if __name__ == "__main__":
    context = nuclio_sdk.Context(logger=logger)
    logger.info("Initializing model...")
    init_model(context, lang="en", device="cpu")
    logger.info("Model initialized successfully.")

    lpcm_bytes, sample_rate = read_audio_to_lpcm_bytes("test/file.wav", sample_rate=16000, bit_depth=16, mono=True)
    
    logger.info("Starting inference handler with chunked processing...")
    # Processa in chunk di 2048 byte (circa 128 ms a 16 kHz con 16-bit mono)
    # Processa in chunk di 4096  byte (circa 256 ms a 16 kHz con 16-bit mono)
    # Processa in chunk di 8192  byte (circa 512 ms a 16 kHz con 16-bit mono)    
    chunk_size = 8192
    results = process_audio_in_chunks(context, lpcm_bytes, chunk_size, sample_rate)    