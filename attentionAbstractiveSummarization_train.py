from data_utils import get_w2c
from data_utils import get_w2i_from_w2c_and_save
from data_utils import get_i2w_from_w2i_and_save
from data_utils import get_data_and_save
from data_utils import load_saved_data
from attentionAbstractiveSummarization_model import AAS
import os
import dynet as dy


def get_data_files(cfg):

    data_dir = cfg.get_data_dir()
    tr_a_file = os.path.join(data_dir, cfg.get_train_article_file())
    tr_t_file = os.path.join(data_dir, cfg.get_train_title_file())
    va_a_file = os.path.join(data_dir, cfg.get_valid_article_file())
    va_t_file = os.path.join(data_dir, cfg.get_valid_title_file())

    generated_data_dir = cfg.get_generated_data_dir()
    w2i_file = os.path.join(generated_data_dir, cfg.get_w2i_file())
    i2w_file = os.path.join(generated_data_dir, cfg.get_i2w_file())
    tr_a_save_file = os.path.join(generated_data_dir, cfg.get_train_article_save_file())
    tr_t_save_file = os.path.join(generated_data_dir, cfg.get_train_title_save_file())
    va_a_save_file = os.path.join(generated_data_dir, cfg.get_valid_article_save_file())
    va_t_save_file = os.path.join(generated_data_dir, cfg.get_valid_title_save_file())

    return tr_a_file, tr_t_file, va_a_file, va_t_file, w2i_file, i2w_file, tr_a_save_file, tr_t_save_file, va_a_save_file, va_t_save_file


def get_loaded_data(cfg, logger):
    tr_a_file, tr_t_file, \
        va_a_file, va_t_file, \
        w2i_file, i2w_file, \
        tr_a_save_file, tr_t_save_file, \
        va_a_save_file, va_t_save_file = get_data_files(cfg)

    w2i = load_saved_data(file_to_load_from=w2i_file, logger=logger)
    i2w = load_saved_data(file_to_load_from=i2w_file, logger=logger)
    assert (len(w2i) == len(i2w))

    tr_articles = load_saved_data(file_to_load_from=tr_a_save_file, logger=logger)
    tr_titles = load_saved_data(file_to_load_from=tr_t_save_file, logger=logger)
    assert (len(tr_articles) == len(tr_titles))

    va_articles = load_saved_data(file_to_load_from=va_a_save_file, logger=logger)
    va_titles = load_saved_data(file_to_load_from=va_t_save_file, logger=logger)
    assert (len(va_articles) == len(va_titles))

    return w2i, tr_articles, tr_titles, va_articles, va_titles


def get_data_from_raw_files(cfg, logger):
    tr_a_file, tr_t_file, \
        va_a_file, va_t_file, \
        w2i_file, i2w_file, \
        tr_a_save_file, tr_t_save_file, \
        va_a_save_file, va_t_save_file = get_data_files(cfg)

    w2c = get_w2c(data_files=(tr_a_file, tr_t_file),
                  logger=logger,
                  max_data_points=cfg.get_max_train_data_points(),
                  min_sent_length=cfg.get_min_sent_length(),
                  max_sent_length=cfg.get_max_sent_length())

    w2i = get_w2i_from_w2c_and_save(w2c=w2c, w2i_file=w2i_file,
                                    bos_sym=cfg.get_bos_sym(),
                                    eos_sym=cfg.get_eos_sym(),
                                    unk_sym=cfg.get_unk_sym(),
                                    logger=logger)

    i2w = get_i2w_from_w2i_and_save(w2i=w2i, i2w_file=i2w_file, logger=logger)
    assert (len(w2i) == len(i2w))

    tr_articles, tr_titles = get_data_and_save(src_data_files=(tr_a_file, tr_t_file),
                                               save_data_files=(tr_a_save_file, tr_t_save_file),
                                               w2i=w2i,
                                               bos_sym=cfg.get_bos_sym(),
                                               eos_sym=cfg.get_eos_sym(),
                                               unk_sym=cfg.get_unk_sym(),
                                               max_data_points=cfg.get_max_train_data_points(),
                                               min_sent_length=cfg.get_min_sent_length(),
                                               max_sent_length=cfg.get_max_sent_length(),
                                               logger=logger)
    assert (len(tr_articles) == len(tr_titles))

    va_articles, va_titles = get_data_and_save(src_data_files=(va_a_file, va_t_file),
                                               save_data_files=(va_a_save_file, va_t_save_file),
                                               w2i=w2i,
                                               bos_sym=cfg.get_bos_sym(),
                                               eos_sym=cfg.get_eos_sym(),
                                               unk_sym=cfg.get_unk_sym(),
                                               max_data_points=cfg.get_max_valid_data_points(),
                                               min_sent_length=cfg.get_min_sent_length(),
                                               max_sent_length=cfg.get_max_sent_length(),
                                               logger=logger)
    assert (len(va_articles) == len(va_titles))

    return w2i, tr_articles, tr_titles, va_articles, va_titles


def get_train_and_valid_data(cfg, logger):
    logger.info("Starting to get training/validation data")
    if cfg.get_should_load_saved_data():
        return get_loaded_data(cfg, logger)
    else:
        return get_data_from_raw_files(cfg, logger)


def train_aas(cfg, logger):
    logger.info("Starting to train")
    w2i, tr_articles, tr_titles, va_articles, va_titles = get_train_and_valid_data(cfg, logger)

    model = dy.Model()
    aas_model = AAS(model=model,
                    vocab_size=len(w2i),
                    word_emb_size=cfg.get_word_emb_size(),
                    context_win_size=cfg.get_context_win_size(),
                    hidden_layer_size=cfg.get_hidden_layer_size())




