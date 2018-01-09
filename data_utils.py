from collections import Counter
import os
import numpy as np
import pickle

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def save_object(object_to_save, file_to_save_in, logger):
    logger.info("Saving object in file = {}".format(file_to_save_in))
    with open(file_to_save_in, 'w+') as f:
        pickle.dump(object_to_save, f)
    logger.info("Done saving object to file")


def load_saved_data(file_to_load_from, logger):
    logger.info("Loading data from file = {}".format(file_to_load_from))
    with open(file_to_load_from, "rb") as f:
        data = pickle.load(f)
    logger.info("Done loading data from file")

    # If the data is iterable, log it's length
    try:
        _ = iter(data)
    except TypeError:
        logger.info("file = {} had non-iterable data".format(file_to_load_from))
    else:
        logger.info("file = {} had non-iterable data of length {}".format(file_to_load_from, len(data) ))

    return data


def are_data_files_ok(data_files, logger):

    if len(data_files) != 2:
        logger.error("data_files has to be of the form (article_file, training_file)")
        raise ValueError

    article_file, title_file = data_files
    if not os.path.exists(article_file):
        logger.error("article_file = {} to create vocab does not exist".format(article_file))
        raise ValueError

    if not os.path.exists(title_file):
        logger.error("article_file = {} to create vocab does not exist".format(title_file))
        raise ValueError

    return True


def are_sentences_ok(a_sent_tokenized, t_sent_tokenized, min_sent_length, max_sent_length):
    if (len(a_sent_tokenized) < min_sent_length) or (len(a_sent_tokenized) > max_sent_length):
        print("a_sent")
        return False

    if (len(t_sent_tokenized) < min_sent_length) or (len(t_sent_tokenized) > max_sent_length):
        print("t_sent")
        return False
    return True


def get_w2c(data_files,
            logger,
            min_sent_length,
            max_sent_length,
            prev_w2c=None,
            max_data_points=None):

    logger.info("Getting w2c")

    if not are_data_files_ok(data_files, logger):
        return

    w2c = prev_w2c
    if w2c is None:
        w2c = Counter()

    article_file, title_file = data_files
    with open(article_file, 'rb') as af, open(title_file, 'rb') as tf:
        for i, (a_line, t_line) in enumerate(zip(af, tf)):
            if (max_data_points is not None) and (i >= max_data_points):
                break

            a_sent_tok = a_line.strip().split()
            t_sent_tok = t_line.strip().split()

            if not are_sentences_ok(a_sent_tok, t_sent_tok, min_sent_length, max_sent_length):
                continue

            w2c.update(a_sent_tok)
            w2c.update(t_sent_tok)
    logger.info("len(w2c) = {}".format(len(w2c)))
    return w2c


def get_w2i_from_w2c_and_save(w2c, w2i_file, unk_sym, bos_sym, eos_sym, logger):
    logger.info("Generating w2i from w2c")
    w2i = {}
    word_id = 0
    for sym in [unk_sym, bos_sym, eos_sym]:
        w2i[sym] = np.int32(word_id)
        word_id += 1
    w2i_from_w2c = {w: np.int32(i + word_id) for i, (w, c) in enumerate(w2c.most_common())}
    w2i.update(w2i_from_w2c)
    save_object(w2i, w2i_file, logger)
    logger.info("len(w2i) = {}".format(len(w2i)))
    return w2i


def get_i2w_from_w2i_and_save(w2i, i2w_file, logger):
    logger.info("Generating i2w from w2i")
    i2w = {i: w for w, i in w2i.iteritems()}
    save_object(i2w, i2w_file, logger)
    logger.info("len(i2w) = {}".format(len(i2w)))
    return i2w


def get_data_and_save(src_data_files, save_data_files,
                      w2i,
                      bos_sym, eos_sym, unk_sym,
                      max_data_points, min_sent_length, max_sent_length,
                      logger):
    logger.info("Getting data and saving")
    if not are_data_files_ok(src_data_files, logger):
        logger.error("Source data files are not ok!")
        return

    for sym in [bos_sym, eos_sym, unk_sym]:
        if sym not in w2i:
            logger.error("sym = {} is not in w2i!".format(sym))
            raise ValueError

    articles = []
    titles = []
    article_file, title_file = src_data_files
    with open(article_file, 'rb') as af, open(title_file, 'rb') as tf:
        for i, (a_line, t_line) in enumerate(zip(af, tf)):
            if (max_data_points is not None) and (i >= max_data_points):
                break

            a_sent_tok_words = a_line.strip().split()
            t_sent_tok_words = t_line.strip().split()

            if not are_sentences_ok(a_sent_tok_words, t_sent_tok_words, min_sent_length, max_sent_length):
                continue

            a_sent_tok_ind = [w2i[w] if w in w2i else w2i[unk_sym] for w in a_sent_tok_words]
            t_sent_tok_ind = [w2i[w] if w in w2i else w2i[unk_sym] for w in t_sent_tok_words]
            t_sent_tok_ind = [w2i[bos_sym]] + t_sent_tok_ind + [w2i[eos_sym]]

            articles.append(a_sent_tok_ind)
            titles.append(t_sent_tok_ind)

    assert (len(articles) == len(titles))

    article_save_file, title_save_file = save_data_files
    save_object(articles, article_save_file, logger)
    save_object(titles, title_save_file, logger)

    logger.info("len(articles) = {}".format(len(articles)))
    logger.info("len(titles) = {}".format(len(titles)))
    return articles, titles
