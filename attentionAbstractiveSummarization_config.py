import ConfigParser


class AASConfig(object):
    def __init__(self):

        self.proj_dir = '/home/shantanu/PycharmProjects/attentionAbstractiveSummarization'
        self.config_file = self.proj_dir + '/attentionAbstractiveSummarization.cfg'

    # Accessors
    def get_cfg(self):
        cfg = ConfigParser.RawConfigParser()
        cfg.read(self.config_file)
        return cfg

    # Accessors/Logging
    def get_log_level(self):
        cfg = self.get_cfg()
        return cfg.get('Logging', 'log_level')

    def get_log_file(self):
        cfg = self.get_cfg()
        return cfg.get('Logging', 'log_file')

    def get_logger(self):
        cfg = self.get_cfg()
        return cfg.get('Logging', 'logger')

    # Accessors/Data
    def get_data_dir(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'data_dir')

    def get_train_article_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'train_article_file')

    def get_train_title_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'train_title_file')

    def get_valid_article_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'valid_article_file')

    def get_valid_title_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'valid_title_file')

    def get_generated_data_dir(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'generated_data_dir')

    def get_w2i_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'w2i_file')

    def get_i2w_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'i2w_file')

    def get_train_article_save_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'train_article_save_file')

    def get_train_title_save_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'train_title_save_file')

    def get_valid_article_save_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'valid_article_save_file')

    def get_valid_title_save_file(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'valid_title_save_file')

    def get_max_train_data_points(self):
        cfg = self.get_cfg()
        if cfg.get('Data', 'max_train_data_points') == "None":
            return None
        else:
            return cfg.getint('Data', 'max_train_data_points')

    def get_max_valid_data_points(self):
        cfg = self.get_cfg()
        if cfg.get('Data', 'max_valid_data_points') == "None":
            return None
        else:
            return cfg.getint('Data', 'max_valid_data_points')

    def get_max_test_data_points(self):
        cfg = self.get_cfg()
        if cfg.get('Data', 'max_test_data_points') == "None":
            return None
        else:
            return cfg.getint('Data', 'max_test_data_points')

    def get_max_sent_length(self):
        cfg = self.get_cfg()
        return cfg.getint('Data', 'max_sent_length')

    def get_min_sent_length(self):
        cfg = self.get_cfg()
        return cfg.getint('Data', 'min_sent_length')

    def get_bos_sym(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'bos_sym')

    def get_eos_sym(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'eos_sym')

    def get_unk_sym(self):
        cfg = self.get_cfg()
        return cfg.get('Data', 'unk_sym')

    def get_should_load_saved_data(self):
        cfg = self.get_cfg()
        return cfg.getboolean('Data', 'should_load_saved_data')

    # Accessors/ExecutionMode
    def get_execution_mode(self):
        cfg = self.get_cfg()
        return cfg.get('ExecutionMode', 'execution_mode')

    # Model
    def get_word_emb_size(self):
        cfg = self.get_cfg()
        return cfg.getint('Model', 'word_emb_size')

    def get_context_win_size(self):
        cfg = self.get_cfg()
        return cfg.getint('Model', 'context_win_size')

    def get_hidden_layer_size(self):
        cfg = self.get_cfg()
        return cfg.getint('Model', 'hidden_layer_size')

    # Training parameters
    def get_minibatch_size(self):
        cfg = self.get_cfg()
        return cfg.getint('TrainingParameters', 'minibatch_size')

    def get_n_epochs(self):
        cfg = self.get_cfg()
        return cfg.getint('TrainingParameters', 'n_epochs')

    def get_n_times_predict_in_epoch(self):
        cfg = self.get_cfg()
        return cfg.getint('TrainingParameters', 'n_times_predict_in_epoch')

    def get_should_save_model_while_training(self):
        cfg = self.get_cfg()
        return cfg.getboolean('TrainingParameters', 'should_save_model_while_training')

    # Manipulators
    def write_sections(self):
        cfg = ConfigParser.RawConfigParser()

        self.set_section_logging(cfg)
        self.set_section_data(cfg)
        self.set_execution_mode(cfg)
        self.set_section_model(cfg)
        self.set_section_training_params(cfg)

        with open(self.config_file, 'wb') as configfile:
            cfg.write(configfile)

    def set_section_logging(self, cfg):
        # Section: Logging
        cfg.add_section('Logging')
        cfg.set('Logging', 'log_level', 'info')
        cfg.set('Logging', 'log_file', self.proj_dir + '/attentionAbstractiveSummarization.log')
        cfg.set('Logging', 'logger', 'AbstractiveSummarization')

    def set_section_data(self, cfg):
        # Section data
        cfg.add_section('Data')
        cfg.set('Data', 'data_dir', self.proj_dir + '/data')
        cfg.set('Data', 'train_article_file', 'train.article.txt')
        cfg.set('Data', 'train_title_file', 'train.title.txt')
        cfg.set('Data', 'valid_article_file', 'valid.article.filter.txt')
        cfg.set('Data', 'valid_title_file', 'valid.title.filter.txt')

        cfg.set('Data', 'generated_data_dir', self.proj_dir + '/generated_data')
        cfg.set('Data', 'w2i_file', 'aas_w2i.txt')
        cfg.set('Data', 'i2w_file', 'aas_i2w.txt')
        cfg.set('Data', 'train_article_save_file', 'train_article_save.txt')
        cfg.set('Data', 'train_title_save_file', 'train_title_save.txt')
        cfg.set('Data', 'valid_article_save_file', 'valid_article_save.txt')
        cfg.set('Data', 'valid_title_save_file', 'valid_title_save.txt')

        cfg.set('Data', 'max_train_data_points', '1')  # Default is None
        cfg.set('Data', 'max_valid_data_points')  # Default is None
        cfg.set('Data', 'max_test_data_points')  # Default is None

        cfg.set('Data', 'max_sent_length', '200')
        cfg.set('Data', 'min_sent_length', '1')

        cfg.set('Data', 'bos_sym', '<s>')
        cfg.set('Data', 'eos_sym', '</s>')
        cfg.set('Data', 'unk_sym', '<unk>')

        cfg.set('Data', 'should_load_saved_data', 'false')

    def set_execution_mode(self, cfg):
        # Execution Mode
        cfg.add_section('ExecutionMode')
        cfg.set('ExecutionMode', 'execution_mode', 'training')

    def set_section_model(self, cfg):
        # Execution Mode
        cfg.add_section('Model')
        cfg.set('Model', 'word_emb_size', '200')
        cfg.set('Model', 'context_win_size', '5')
        cfg.set('Model', 'hidden_layer_size', '400')

    def set_section_training_params(self, cfg):
        # Training parameters
        cfg.add_section('TrainingParameters')
        cfg.set('TrainingParameters', 'minibatch_size', '16')
        cfg.set('TrainingParameters', 'n_epochs', '2')
        cfg.set('TrainingParameters', 'n_times_predict_in_epoch', '2')
        cfg.set('TrainingParameters', 'should_save_model_while_training', 'true')

