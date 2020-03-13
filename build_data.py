from config import Config
import xlrd, os, random, csv
from data_utils import AnnotationDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word
from general_utils import clean_word

def build_data(config):
    annotations = []
    meta_filename = 'sw%s%s-ms98-a-trans.text' # % (file_id, speaker_id)
    # print(meta_filename)
    for idx in os.listdir(config.wimp_corpus):
        idx_path = os.path.join(config.wimp_corpus, idx)
        if os.path.isfile(idx_path):
            continue

        for file_id in os.listdir(idx_path):
            folder = os.path.join(idx_path, file_id)
            if os.path.isfile(folder):
                continue

            fileName = os.listdir(folder)
            # wimp_trans_files = [os.path.join(folder, meta_filename % (file_id, 'A')),
            # os.path.join(folder, meta_filename % (file_id, 'B'))]
            # wimp_trans_files = [folder +'/'+x os.listdir(folder)
            wimp_trans_files = []
            for x in fileName:
                wimp_trans_files.append(folder+'/'+x+'/sw'+x+'A-ms98-a-trans.text')
                wimp_trans_files.append(folder+'/'+x+'/sw'+x+'B-ms98-a-trans.text')

            fileName = os.listdir(os.path.join(config.swd_transcripts, file_id))
            var = os.path.join(config.swd_transcripts, file_id)
            swd_trans_files = []
            for x in fileName:
                swd_trans_files.append(var + '/'+x+'/sw'+x+'A-ms98-a-trans.text')
                swd_trans_files.append(var + '/'+x+'/sw'+x+'B-ms98-a-trans.text')

            for i, wimp_trans_file in enumerate(wimp_trans_files):
                swd_trans_file = swd_trans_files[i]
                file_id, speaker = swd_trans_file.split("/")[-2:]
                speaker = speaker[6]
                
                try:
                    with open(wimp_trans_file) as w_file_obj, open(swd_trans_file) as s_file_obj:
                    # print(wimp_trans_file[34:41],swd_trans_file[34:41])
                        if wimp_trans_file[34:41] == swd_trans_file[34:41]:
                            for line_num, (anns_, wrds_) in enumerate(zip(w_file_obj, s_file_obj)):
                                sentence = []
                                anns = anns_.strip().split(' ')[3:]
                                wrds  = wrds_.strip().split(' ')[3:]
                                try:
                                    assert(len(anns) == len(wrds)), \
                                    "file mismatch, line %d : %s and %s" % (line_num, swd_trans_file, wimp_trans_file)

                                    for id_, wrd in enumerate(wrds):
                                        wrd = clean_word(wrd)
                                        if wrd != '':
                                            sentence.append([(file_id, line_num, speaker), wrd, float(anns[id_])])
                                    
                                    if len(sentence) != 0:
                                        annotations.append(sentence)
                                except:
                                    pass
                except:
                    pass
                
    
    random.shuffle(annotations)
    # print(len(annotations))
    #80% for training, 10% dev, 10% test
    # print(annotations)
    d_train = annotations[ : 140]
    d_test = annotations[140 : 150]
    d_dev = annotations[150:160 ]
    def prep_text_data(D, outfile):
        with open(outfile, 'w') as f:
            for sent in D:
                for _, word, label in sent:
                    f.write("%s %f\n" % (word, label))
                f.write("\n")

    prep_text_data(d_train, config.train_filename)
    prep_text_data(d_test, config.test_filename)
    prep_text_data(d_dev, config.dev_filename)

    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = AnnotationDataset(config.dev_filename, processing_word)
    test  = AnnotationDataset(config.test_filename, processing_word)
    train = AnnotationDataset(config.train_filename, processing_word)

    # Build Word and Tag vocab
    # Vocabulary is built using training data
    vocab_words, vocab_tags = get_vocabs([train])
    vocab_glove = get_glove_vocab(config.glove_filename)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.words_filename)
    write_vocab(vocab_tags, config.tags_filename)

    # Trim GloVe Vectors
    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename, 
                                config.trimmed_filename, config.dim)

    # Build and save char vocab
    train = AnnotationDataset(config.train_filename)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.chars_filename)


if __name__ == "__main__":
    config = Config()
    build_data(config)