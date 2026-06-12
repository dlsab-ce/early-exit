import logging
import time
import nuclio_sdk

from util.audio_utils import read_audio_to_lpcm_bytes
from inference_handler import init_model, handler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s - %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def process_audio_in_chunks(context, lpcm_bytes, chunk_size, sample_rate=16000):
    """
    Divide l'array di byte LPCM in chunk e invoca il handler per ogni chunk.
    Introduce un delay tra i chunk per simulare uno streaming real time.
    
    Args:
        context: Contesto nuclio
        lpcm_bytes (bytes): Array di byte LPCM
        chunk_size (int): Numero di byte per chunk
        sample_rate (int): Frequenza di campionamento (per logging e calcolo delay)
    
    Esempio:
        >>> process_audio_in_chunks(context, lpcm_bytes, chunk_size=2048, sample_rate=16000)
    """
    num_chunks = (len(lpcm_bytes) + chunk_size - 1) // chunk_size
    logger.info(f"Processing {len(lpcm_bytes)} bytes in {num_chunks} chunks of {chunk_size} bytes each")
    
    # Calcola il delay in secondi: numero di campioni nel chunk / sample rate
    # chunk_size è in byte, ogni campione è 2 byte (16-bit PCM)
    samples_per_chunk = chunk_size // 2
    chunk_duration = samples_per_chunk / sample_rate
    logger.info(f"Chunk duration: {chunk_duration:.3f}s (for real-time streaming simulation)")
    
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
        
        # Aggiungi delay per simulare streaming real time (tranne per l'ultimo chunk)
        if chunk_num < num_chunks:
            time.sleep(chunk_duration)
    
    logger.info(f"All {num_chunks} chunks processed")
    return results

if __name__ == "__main__":
    context = nuclio_sdk.Context(logger=logger)
    logger.info("Initializing model...")
    init_model(context, lang="en", device="cpu")
    logger.info("Model initialized successfully.")

    lpcm_bytes, sample_rate = read_audio_to_lpcm_bytes("test/file.wav", sample_rate=16000, bit_depth=16, mono=True)
    
    logger.info("Starting inference handler with chunked processing...")
    # Processa in chunk di 4096 byte (circa 128 ms a 16 kHz con 16-bit mono)
    # Processa in chunk di 8192  byte (circa 256 ms a 16 kHz con 16-bit mono)
    # Processa in chunk di 16384 byte (circa 512 ms a 16 kHz con 16-bit mono)   
    # Processa in chunk di 32768 byte (circa 1 secondo a 16 kHz con 16-bit mono) 
    chunk_size = 16384
    results = process_audio_in_chunks(context, lpcm_bytes, chunk_size, sample_rate)    