from attentionAbstractiveSummarization_config import AASConfig
from attentionAbstractiveSummarization_train import train_aas
import logging


def get_logger(cfg):
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename=cfg.get_log_file(),
                        level=logging.INFO)
    logger = logging.getLogger(cfg.get_logger())
    return logger


def setup_testing(cfg, logger):
    logger.info("testing currently not yet setup")
    pass


def main():
    cfg = AASConfig()
    cfg.write_sections()

    logger = get_logger(cfg)
    logger.info("in main")

    if cfg.get_execution_mode() == "training":
        train_aas(cfg, logger)
    elif cfg.get_execution_mode() == "testing":
        setup_testing(cfg, logger)
    else:
        logger.error("Unsupported mode")

if __name__ == "__main__":
    main()