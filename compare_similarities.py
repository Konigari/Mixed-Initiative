import glob
import os
from swda.swda import CorpusReader

# from bert_alternate.bert_similarity import BertSimilarity

# similarity_processor = BertSimilarity()

QUESTIONS = ["qy", "qw", "qo", "qr"]

def get_similarity(sentence1, sentence2):
    print(sentence1, sentence2)
    return 1
    # similarity_processor.get_similarity(sentence1, sentence2)


def is_question(utterance):
    return utterance.damsl_act_tag() in QUESTIONS


if __name__ == '__main__':
    data_dir = "swda/swda"
    scan_range = 5
    cr = CorpusReader("swda1/swda")
    for dialog in cr.iter_transcripts(display_progress=True):
            for index, utterance in enumerate(dialog.utterances):
                if index < scan_range or index >= len(dialog.utterances) - scan_range:
                    continue
                if is_question(utterance):
                    for i in range(-5, 5):
                        print(get_similarity(utterance.text, dialog.utterances[index+i].text))
    # pyplot.bar(range(scan_range*2), aggregate)
