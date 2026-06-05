import logging
import nuclio_sdk

from inference_handler import init_model, handler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s - %(name)s: %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    context = nuclio_sdk.Context(logger=logger)
    logger.info("Initializing model...")
    model, inf = init_model(context, lang="en", device="cpu")
    logger.info("Model initialized successfully.")
    
    #logger.info("Starting inference handler...")
    #handler(model, inf)    