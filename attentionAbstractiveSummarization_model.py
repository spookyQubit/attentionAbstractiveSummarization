import dynet as dy


class AAS(object):
    def __init__(self,
                 model,
                 vocab_size,
                 word_emb_size,
                 context_win_size,
                 hidden_layer_size):

        """
        :param model: dynet model
        :param vocab_size: the size of w2i
        :param word_emb_size: the size of word emb
        :param context_win_size: context window size. Used only for y. Same as c.
        :param hidden_layer_size: hidden layer size

        This way of structuring the class (specifically, passing model as an argument to the class)
        is similar to what Yoav Goldberg has in his presentation.

        These parameters are used to build the following model parameters:
        p(y_{i+1}| y_{c}, x; E,U,V,W) ~ exp{Vh + W*emb(x, y_{c})}
                        \tilda{y}_{c} = [Ey_{i-c+1}, ..., Ey_{i}]
                                   h  = tanh(U\tilda{y}_{c})

        ---------------------------------------------------------------------------
          Parameters           |          Shapes
        ---------------------------------------------------------------------------
              E                | (word_emb_size, vocab_size)
              \tilda{y}_{c}    | (context_win_size*word_emb_size, 1)
              U                | (hidden_layer_size, context_win_size*word_emb_size)
              h                | (hidden_layer_size, 1)
              V                | (vocab_size, hidden_layer_size)
              W                | (vocab_size, hidden_layer_size)
              emb(x, y_{c})    | (hidden_layer_size, 1)
        ---------------------------------------------------------------------------
        """

        self.vocab_size = vocab_size
        self.word_emb_size = word_emb_size
        self.context_win_size = context_win_size
        self.hidden_layer_size = hidden_layer_size

        self.E = model.add_lookup_parameters((self.word_emb_size, self.vocab_size))
        self.U = model.add_parameters((self.hidden_layer_size, self.context_win_size * self.word_emb_size))
        self.V = model.add_parameters((self.vocab_size, self.hidden_layer_size))

        # Encoder
        self.W = model.add_parameters((self.vocab_size, self.hidden_layer_size))

    def __call__(self, in_exp):
        """
        :param in_exp:
        :return:
        """

