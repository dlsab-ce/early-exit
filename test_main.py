import logging
import nuclio_sdk

from util.audio_utils import read_audio_to_lpcm_bytes
from inference_handler import init_model, handler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s - %(name)s: %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    context = nuclio_sdk.Context(logger=logger)
    logger.info("Initializing model...")
    init_model(context, lang="en", device="cpu")
    logger.info("Model initialized successfully.")

    lpcm_bytes, sample_rate = read_audio_to_lpcm_bytes("test/file.wav", sample_rate=16000, bit_depth=16)
    
    #logger.info("Starting inference handler...")
    #handler(model, inf)    