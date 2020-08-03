class config:
    def __init__(self):
        self.class_size = 2
        self.char_embedding_len = 100
        self.word_embedding_len = 100

        self.max_char_len = 15
        self.max_word_len = 15

        self.batch_size = 1000

        self.char_vocab_len = 1692

        self.learning_rate = 0.0002

        self.keep_prob_ae = 0.8
        self.keep_prob_fully = 0.8
        self.keep_prob_embed = 0.5

        self.epochs = 100

        self.lstm_hidden = 100
