
# Modified from https://github.com/karpathy/char-rnn
# This version is for cases where one has already segmented train/val/test splits
import codecs
import numpy as np
from os import path
import gc
import re
from collections import Counter, OrderedDict, namedtuple

Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])
ENCODING="iso-8859-15"
VECTOR_SIZE=20

def vocab_unpack(vocab):
    return vocab['idx2word'], vocab['word2idx'], vocab['idx2char'], vocab['char2idx']


def process_line(line):
    lsplit = line.split()
    assert(len(lsplit) == VECTOR_SIZE+5)
    filename = lsplit[0]
    spk = lsplit[1]
    word = lsplit[4]
    prob = map(float, lsplit[5:VECTOR_SIZE+5])
    return filename, word, spk, prob


class BatchLoaderUnk:
    def __init__(self, tokens, data_dir, batch_size, seq_length, max_word_l, n_words, n_chars, delay):
        self.n_words = n_words
        self.n_chars = n_chars
        train_file = path.join(data_dir, 'train.ctm')
        valid_file = path.join(data_dir, 'valid.ctm')
        test_file = path.join(data_dir, 'test.ctm')
        input_files = [train_file, valid_file, test_file]
        vocab_file = path.join(data_dir, 'vocab.npz')
        tensor_file = path.join(data_dir, 'data')
        char_file = path.join(data_dir, 'data_char')
        spk_file = path.join(data_dir, 'spk')
        prb_file = path.join(data_dir, 'prb')

        # construct a tensor with all the data
        if not path.exists(vocab_file):
            print 'one-time setup: preprocessing input train/valid/test files in dir: ', data_dir
            self.text_to_tensor(tokens, input_files, vocab_file, tensor_file, char_file, spk_file, prb_file, max_word_l)

        print('loading data files...')
        all_data = []
        all_data_char = []
        all_spk = []
        all_prb = []
        for split in range(3):
            all_data.append(np.load("{}_{}.npy".format(tensor_file, split)))  # train, valid, test tensors
            all_data_char.append(np.load("{}_{}.npy".format(char_file, split)))  # train, valid, test character indices
            all_spk.append(np.load("{}_{}.npy".format(spk_file, split)))  # train, valid, test spk indices
            all_prb.append(np.nan_to_num(np.load("{}_{}.npy".format(prb_file, split))))  # train, valid, test acoustic vectors
        vocab_mapping = np.load(vocab_file)
        self.idx2word, self.word2idx, self.idx2char, self.char2idx = vocab_unpack(vocab_mapping)
        self.vocab_size = len(self.idx2word)
        print 'Word vocab size: %d, Char vocab size: %d' % (len(self.idx2word), len(self.idx2char))
        # create word-char mappings
        self.max_word_l = all_data_char[0].shape[1]
        # cut off the end for train/valid sets so that it divides evenly
        # test set is not cut off
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.data_sizes = []
        self.split_sizes = []
        self.all_batches = []
        print('reshaping tensors...')
        for split, data in enumerate(all_data):
            data_len = data.shape[0]
            self.data_sizes.append(data_len)
            if split < 2 and data_len % (batch_size * seq_length) != 0:
                data = data[:batch_size * seq_length * (data_len // (batch_size * seq_length))]
            ydata = all_spk[split][:len(data)]
            if delay != 0:
                ydata[delay:] = ydata[:-delay]
            spk = np.ones_like(ydata)
            spk[1:] = ydata[:-1]
            # prb = np.ones(len(data), dtype='float32')
            prb = all_prb[split][:len(data)]
            print 'split: {}. Accuracy: {}'.format(split, np.mean(spk == ydata))
            data_char = all_data_char[split][:len(data)]
            if split < 2:
                rdata = data.reshape((batch_size, -1))
                rydata = ydata.reshape((batch_size, -1))
                rspk = spk.reshape((batch_size, -1))
                rprb = prb.reshape((batch_size, -1, VECTOR_SIZE))
                rdata_char = data_char.reshape((batch_size, -1, self.max_word_l))
            else: # for test we repeat dimensions to batch size (easier but inefficient evaluation)
                nseq = (data_len + (seq_length - 1)) // seq_length
                rdata = data.copy()
                rdata.resize((1, nseq*seq_length))
                rdata = np.tile(rdata, (batch_size, 1))
                rydata = ydata.copy()
                rydata.resize((1, nseq*seq_length))
                rydata = np.tile(rydata, (batch_size, 1))
                rspk = spk.copy()
                rspk.resize((1, nseq*seq_length))
                rspk = np.tile(rspk, (batch_size, 1))
                rprb = prb.copy()
                rprb.resize((1, nseq*seq_length, VECTOR_SIZE))
                rprb = np.tile(rprb, (batch_size, 1, 1))
                rdata_char = data_char.copy()
                rdata_char.resize((1, nseq*seq_length, rdata_char.shape[1]))
                rdata_char = np.tile(rdata_char, (batch_size, 1, 1))
            # split in batches
            x_batches = np.split(rdata, rdata.shape[1]/seq_length, axis=1)
            y_batches = np.split(rydata, rydata.shape[1]/seq_length, axis=1)
            s_batches = np.split(rspk, rspk.shape[1]/seq_length, axis=1)
            p_batches = np.split(rprb, rprb.shape[1]/seq_length, axis=1)
            x_char_batches = np.split(rdata_char, rdata_char.shape[1]/seq_length, axis=1)
            nbatches = len(x_batches)
            self.split_sizes.append(nbatches)
            assert len(x_batches) == len(y_batches)
            assert len(x_batches) == len(s_batches)
            assert len(x_batches) == len(x_char_batches)
            self.all_batches.append((x_batches, y_batches, s_batches, x_char_batches, p_batches))

        self.batch_idx = [0,0,0]
        self.word_vocab_size = len(self.idx2word)
        print 'data load done. Number of batches in train: %d, val: %d, test: %d' \
              % (self.split_sizes[0], self.split_sizes[1], self.split_sizes[2])
        gc.collect()

    def reset_batch_pointer(self, split_idx, batch_idx=0):
        self.batch_idx[split_idx] = batch_idx

    def next_batch(self, split_idx):
        while True:
            # split_idx is integer: 0 = train, 1 = val, 2 = test
            idx = self.batch_idx[split_idx]
            self.batch_idx[split_idx] += 1
            if self.batch_idx[split_idx] >= self.split_sizes[split_idx]:
                self.batch_idx[split_idx] = 0 # cycle around to beginning

            # pull out the correct next batch
            word = self.all_batches[split_idx][0][idx]
            word[word >= self.n_words] = 0
            ydata = self.all_batches[split_idx][1][idx].astype('float32')
            spk = self.all_batches[split_idx][2][idx].astype('float32')
            chars = self.all_batches[split_idx][3][idx]
            prb = self.all_batches[split_idx][4][idx]
            # expand dims
            ydata = np.expand_dims(ydata, axis=2)
                    
            yield ({'word':word, 'chars':chars, 'spk':spk, 'prb':prb}, ydata)

    def text_to_tensor(self, tokens, input_files, out_vocabfile, out_tensorfile, out_charfile, out_spkfile, out_prbfile, max_word_l):
        print 'Processing text into tensors...'
        max_word_l_tmp = 0 # max word length of the corpus
        idx2word = [tokens.UNK] # unknown word token
        word2idx = OrderedDict()
        word2idx[tokens.UNK] = 0
        idx2char = [tokens.ZEROPAD, tokens.START, tokens.END, tokens.UNK] # zero-pad, start-of-word, end-of-word tokens
        char2idx = OrderedDict()
        char2idx[tokens.ZEROPAD] = 0
        char2idx[tokens.START] = 1
        char2idx[tokens.END] = 2
        char2idx[tokens.UNK] = 3
        split_counts = []

        # first go through train/valid/test to get max word length
        # if actual max word length is smaller than specified
        # we use that instead. this is inefficient, but only a one-off thing so should be fine
        # also counts the number of tokens
        wordcount = Counter()
        charcount = Counter()
        for	split in range(3): # split = 0 (train), 1 (val), or 2 (test)

            def update(word):
                if word == 'VOX':
                    word = tokens.UNK # replace VOX with a single character
                if word.startswith('<'):
                    word = tokens.EOS # replace <s> and </s> with a single character

                if word[0] == tokens.UNK:
                    if len(word) > 1: # unk token with character info available
                        word = word[2:]
                else:
                    wordcount.update([word])
                word = word.replace(tokens.UNK, '')
                charcount.update(word)
            
            f = codecs.open(input_files[split], 'r', encoding=ENCODING)
            counts = 0
            for line in f:
                _, word, _, _ = process_line(line)
                update(word)
                max_word_l_tmp = max(max_word_l_tmp, len(word) + 2) # add 2 for start/end chars
                counts += 1
            f.close()
            split_counts.append(counts)

        print 'Most frequent words:', len(wordcount)
        for ii, ww in enumerate(wordcount.most_common(self.n_words - 1)):
            word = ww[0]
            word2idx[word] = ii + 1
            idx2word.append(word)
            if ii < 3: print word

        print 'Most frequent chars:', len(charcount)
        for ii, cc in enumerate(charcount.most_common(self.n_chars - 4)):
            char = cc[0]
            char2idx[char] = ii + 4
            idx2char.append(char)
            if ii < 3: print char

        print 'Char counts:'
        for ii, cc in enumerate(charcount.most_common()):
            print ii, cc[0].encode(ENCODING), cc[1]
                    
        print 'After first pass of data, max word length is: ', max_word_l_tmp
        print 'Token count: train %d, val %d, test %d' % (split_counts[0], split_counts[1], split_counts[2])

        # if actual max word length is less than the limit, use that
        max_word_l = min(max_word_l_tmp, max_word_l)

        for split in range(3):  # split = 0 (train), 1 (val), or 2 (test)
            # Preallocate the tensors we will need.
            # Watch out the second one needs a lot of RAM.
            output_tensor = np.empty(split_counts[split], dtype='int32')
            output_chars = np.zeros((split_counts[split], max_word_l), dtype='int32')
            output_spk = np.ones(split_counts[split], dtype='int32')
            output_prb = np.empty((split_counts[split], VECTOR_SIZE), dtype='float32')

            def append(word, word_num, first_word, prb):
                if word == 'VOX':
                    word = tokens.UNK # replace VOX with a single character
                if word.startswith('<'):
                    word = tokens.EOS # replace <s> and </s> with a single character

                chars = [char2idx[tokens.START]] # start-of-word symbol
                if word[0] == tokens.UNK and len(word) > 1: # unk token with character info available
                    word = word[2:]
                    output_tensor[word_num] = word2idx[tokens.UNK]
                else:
                    output_tensor[word_num] = word2idx[word] if word in word2idx else word2idx[tokens.UNK]
                chars += [char2idx[char] for char in word if char in char2idx]
                chars.append(char2idx[tokens.END]) # end-of-word symbol
                if len(chars) >= max_word_l:
                    chars[max_word_l-1] = char2idx[tokens.END]
                    output_chars[word_num] = chars[:max_word_l]
                else:
                    output_chars[word_num, :len(chars)] = chars
                output_spk[word_num] = first_word
                output_prb[word_num] = prb
                return word_num + 1

            f = codecs.open(input_files[split], 'r', encoding=ENCODING)
            word_num = 0
            prev_filename = ''
            prev_spk = ''
            for line in f:
                filename, word, spk, prb = process_line(line)
                if (filename != prev_filename) or (spk != prev_spk):
                    first_word = 1;
                    prev_filename = filename
                    prev_spk = spk
                else:
                    first_word = 0;
                word_num = append(word, word_num, first_word, prb)
            assert(word_num == split_counts[split])
            f.close()
            tensorfile_split = "{}_{}.npy".format(out_tensorfile, split)
            print 'saving ', tensorfile_split
            np.save(tensorfile_split, output_tensor)
            charfile_split = "{}_{}.npy".format(out_charfile, split)
            print 'saving ', charfile_split
            np.save(charfile_split, output_chars)
            spkfile_split = "{}_{}.npy".format(out_spkfile, split)
            print 'saving ', spkfile_split
            np.save(spkfile_split, output_spk)
            prbfile_split = "{}_{}.npy".format(out_prbfile, split)
            print 'saving ', prbfile_split
            np.save(prbfile_split, output_prb)

        # save output preprocessed files
        print 'saving ', out_vocabfile
        np.savez(out_vocabfile, idx2word=idx2word, word2idx=word2idx, idx2char=idx2char, char2idx=char2idx)
