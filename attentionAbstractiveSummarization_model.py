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

        BOW enc:
                        enc(x, y_{c}) = avg([Fy_{i-c+1}, ..., Fy_{i}])

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
              F                | (hidden_layer_size, vocab_size)
        ---------------------------------------------------------------------------
        """

        self.vocab_size = vocab_size
        self.word_emb_size = word_emb_size
        self.context_win_size = context_win_size
        self.hidden_layer_size = hidden_layer_size

        self._E = model.add_lookup_parameters((self.word_emb_size,
                                               self.vocab_size))
        self.E = None
        self._U = model.add_parameters((self.hidden_layer_size,
                                        self.context_win_size * self.word_emb_size))
        self.U = None
        self._V = model.add_parameters((self.vocab_size,
                                        self.hidden_layer_size))
        self.V = None

        # Encoder
        self._W = model.add_parameters((self.vocab_size,
                                        self.hidden_layer_size))
        self.W = None
        self._F = model.add_parameters((self.hidden_layer_size,
                                        self.vocab_size))
        self.F = None

    def __call__(self, article, title):
        pass

    def expressions_to_parameters(self):

        self.E = dy.parameter(self._E)
        self.U = dy.parameter(self._U)
        self.V = dy.parameter(self._V)
        self.W = dy.parameter(self._W)
        self.F = dy.parameter(self._F)



