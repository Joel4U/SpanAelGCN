# 
# @author: Allan
#

from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
import itertools
from transformers import PreTrainedTokenizerFast, RobertaTokenizer, AutoTokenizer
import numpy as np
from src.config.config import PaserModeType, DepModelType
from src.data.data_utils import convert_iobes, build_spanlabel_idx, build_label_idx, build_deplabel_idx, enumerate_spans
from src.data import Instance
import logging
import unicodedata
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)

class TransformersNERDataset(Dataset):
    def __init__(self, parser_mode: int, dep_model: int, file: str,
                 tokenizer: AutoTokenizer, # tokenizer: PreTrainedTokenizerFast,
                 is_train: bool,
                 sents: List[List[str]] = None,
                 label2idx: Dict[str, int] = None,
                 deplabel2idx: Dict[str, int] = None,
                 number: int = -1):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        ## read all the instances. sentences and labels
        self.parser_mode = parser_mode
        self.dep_mode = dep_model
        self.max_entity_length = 0
        self.insts = self.read_file(file=file, number=number) if sents is None else self.read_from_sentences(sents)
        minus = int((self.max_entity_length + 1) * self.max_entity_length / 2)
        self.max_num_span = 128 * self.max_entity_length - minus # self.max_length = 128 max length of dataset
        if is_train:
            if label2idx is not None:
                print(f"[WARNING] YOU ARE USING EXTERNAL label2idx, WHICH IS NOT BUILT FROM TRAINING SET.")
                self.label2idx = label2idx
            else:
                print(f"[Data Info] Using the training set to build label index")
                if parser_mode == PaserModeType.crf:
                    idx2labels, label2idx = build_label_idx(self.insts)
                else:
                    idx2labels, label2idx = build_spanlabel_idx(self.insts)

                self.idx2labels = idx2labels
                self.label2idx = label2idx
                if dep_model != DepModelType.none:
                    self.deplabel2idx, self.root_dep_label_id = build_deplabel_idx(self.insts)
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            if dep_model != 0:
                self.deplabel2idx = deplabel2idx

            # check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)
        if dep_model != DepModelType.none:
            self.insts_ids = self.convert_instances_to_feature_tensors(parser_mode, self.insts, tokenizer, self.deplabel2idx, label2idx)
        else:
            self.insts_ids = self.convert_instances_to_feature_tensors(parser_mode, self.insts, tokenizer, None, label2idx)
        self.tokenizer = tokenizer

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def convert_instances_to_feature_tensors(self, parser_mode: int, instances: List[Instance],
                                             tokenizer: AutoTokenizer, #PreTrainedTokenizerFast,
                                             deplabel2idx: Dict[str, int],
                                             label2idx: Dict[str, int]) -> List[Dict]:
        features = []
        print("[Data Info] We are not limiting the max length in tokenizer. You should be aware of that")

        for idx, inst in tqdm(enumerate(instances)):
            words = inst.ori_words
            orig_to_tok_index = []
            # res = tokenizer.encode_plus(words, is_split_into_words=True) # RobertaTokenizerFast
            # subword_idx2word_idx = res.word_ids(batch_index=0) # RobertaTokenizerFast
            input_ids = tokenizer.encode(words, is_split_into_words=True)
            attention_mask = [1] * len(input_ids)
            tokens = [tokenizer.tokenize(w) for w in words]
            subword_idx2word_idx = [None] + list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)])) + [None]
            prev_word_idx = -1
            for i, mapped_word_idx in enumerate(subword_idx2word_idx):
                """
                Note: by default, we use the first wordpiece/subword token to represent the word
                If you want to do something else (e.g., use last wordpiece to represent), modify them here.
                """
                if mapped_word_idx is None:  ## cls and sep token
                    continue
                if mapped_word_idx != prev_word_idx:
                    ## because we take the first subword to represent the whold word
                    orig_to_tok_index.append(i)
                    prev_word_idx = mapped_word_idx
            assert len(orig_to_tok_index) == len(words)

            # segment_ids = [0] * len(res["input_ids"]) # RobertaTokenizerFast
            segment_ids = [0] * len(input_ids)
            if deplabel2idx != None:
                dep_labels = inst.dep_labels
                deplabel_ids = [deplabel2idx[dep_label] for dep_label in dep_labels] if dep_labels else [-100] * len(words)
                dephead_ids = inst.dep_heads
            else:
                deplabel_ids = None
                dephead_ids = None
            if parser_mode == PaserModeType.crf:
                labels = inst.labels
                label_ids = [label2idx[label] for label in labels] if labels else [-100] * len(words)
                features.append({"input_ids": input_ids, # res["input_ids"], # RobertaTokenizerFast
                                 "attention_mask": attention_mask, #res["attention_mask"], # RobertaTokenizerFast
                                 "orig_to_tok_index": orig_to_tok_index,
                                 "token_type_ids": segment_ids,
                                 "word_seq_len": len(orig_to_tok_index),
                                 "dephead_ids": dephead_ids,
                                 "deplabel_ids": deplabel_ids,
                                 "label_ids": label_ids})
            else:
                span_labels = {spanlabel[0]: label2idx[spanlabel[1]] for spanlabel in inst.span_labels}
                span_lens = []
                span_weights = []
                # If entity_labels is empty, assign default "O" label for the entire sentence
                max_span_length = min(self.max_entity_length, len(words))
                spanlabel_ids = []
                for entity_start in range(len(words)):
                    for entity_end in range(entity_start, entity_start + max_span_length):
                        if entity_end < len(words):
                            label = span_labels.get((entity_start, entity_end), 0)
                            weight = 0.5  # self.neg_span_weight = 0.5
                            if label != 0:  # 0 is label 'O'
                                weight = 1.0
                            span_weights.append(weight)
                            spanlabel_ids.append(((entity_start, entity_end), label))
                            span_lens.append(entity_end - entity_start + 1)
                spans = []
                for start, end in self.enumerate_spans(words, max_span_width=max_span_length):
                    spans.append((start, end))

                features.append({"input_ids": input_ids, # res["input_ids"], # RobertaTokenizerFast,
                                 "attention_mask": attention_mask, # res["attention_mask"],# RobertaTokenizerFast,
                                 "orig_to_tok_index": orig_to_tok_index, "token_type_ids": segment_ids,
                                 "word_seq_len": len(orig_to_tok_index),
                                 "dephead_ids": dephead_ids, "deplabel_ids": deplabel_ids,
                                 "span_lens": span_lens, "span_weight": span_weights,
                                 "span_mask": [1] * len(span_weights), "spanlabel_ids": spanlabel_ids})
        return features

    def read_from_sentences(self, sents: List[List[str]]):
        """
        sents = [['word_a', 'word_b'], ['word_aaa', 'word_bccc', 'word_ccc']]
        """
        insts = []
        for sent in sents:
            insts.append(Instance(words=sent, ori_words=sent))
        return insts

    def get_chunk_type(self, tok):
        tag_class = tok.split('-')[0]
        tag_type = '-'.join(tok.split('-')[1:])
        return tag_class, tag_type

    def get_chunks(self, seq):
        default = 'O'
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = ((chunk_start, i-1), chunk_type)
                chunks.append(chunk)
                if (i - chunk_start) > self.max_entity_length:
                    self.max_entity_length = (i - chunk_start)
                chunk_type, chunk_start = None, None
            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = ((chunk_start, i-1), chunk_type)
                    if (i - chunk_start) > self.max_entity_length:
                        self.max_entity_length = (i - chunk_start)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = ((chunk_start, len(seq)-1), chunk_type)
            if len(seq) - chunk_start > self.max_entity_length:
                self.max_entity_length = len(seq) - chunk_start
            chunks.append(chunk)

        return chunks

    def enumerate_chunk(self, labels):
        entity_infos = []
        entity_labels = self.get_chunks(labels)
        if not entity_labels:  # If entity_labels is empty, assign default "O" label for the entire sentence
            entity_infos.append(((0, len(labels)), 0, (0, len(labels) - 1)))
        else:
            for entity_start in range(len(labels)):
                doc_entity_start = entity_start
                if doc_entity_start not in range(len(labels)):
                    continue
                for entity_end in range(entity_start + 1, len(labels) + 1):
                    doc_entity_end = entity_end
                    if doc_entity_end not in range(len(labels)):
                        continue
                    label = entity_labels.get((doc_entity_start, doc_entity_end), 0)
                    entity_infos.append(((entity_start + 1, entity_end), label, (doc_entity_start, doc_entity_end - 1)))


    def enumerate_spans(self, sentence, max_span_width, min_span_width=1):

        max_span_width = max_span_width or len(sentence)
        spans = []

        for start_index in range(len(sentence)):
            last_end_index = min(start_index + max_span_width, len(sentence))
            first_end_index = min(start_index + min_span_width - 1, len(sentence))
            for end_index in range(first_end_index, last_end_index):
                start = start_index
                end = end_index
                spans.append((start, end))
        return spans

    def read_file(self, file: str, number: int = -1) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}")
        print(f"[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements")
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            dep_heads = []
            dep_labels = []
            ori_words = []
            labels = []
            chunks = []
            find_root = False
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    continue
                if line == "" and len(words) != 0:
                    if self.parser_mode == PaserModeType.crf:
                        labels = convert_iobes(labels)
                    else:
                        chunks = self.get_chunks(labels)
                    if 'conll' in file:
                        insts.append(Instance(words=words, ori_words=ori_words, dep_heads=None, dep_labels=None, span_labels=chunks, labels=labels))
                    else:
                        insts.append(Instance(words=words, ori_words=ori_words, dep_heads=dep_heads, dep_labels=dep_labels, span_labels=chunks, labels=labels))
                        find_root = False
                        dep_heads = []
                        dep_labels = []
                    words = []
                    ori_words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                elif line == "" and len(words) == 0:
                    continue
                ls = line.split()
                if 'conll' in file:
                    word, label = ls[0], ls[-1]
                else:
                    word, head, dep_label, label = ls[1], int(ls[6]), ls[7], ls[-1]
                    if head == 0 and find_root:
                        raise ValueError("already have a root")
                    dep_heads.append(head - 1) ## because of 0-indexed.
                    dep_labels.append(dep_label)
                ori_words.append(word)
                words.append(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_to_max_length(self, batch:List[Dict]):
        word_seq_len = [len(feature["orig_to_tok_index"]) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])
        if self.parser_mode == PaserModeType.span:
            max_span_num = max([len(feature["spanlabel_ids"]) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature["input_ids"])
            input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            mask = feature["attention_mask"] + [0] * padding_length
            type_ids = feature["token_type_ids"] + [self.tokenizer.pad_token_type_id] * padding_length
            padding_word_len = max_seq_len - len(feature["orig_to_tok_index"])
            orig_to_tok_index = feature["orig_to_tok_index"] + [0] * padding_word_len
            if self.dep_mode != DepModelType.none:
                dephead_ids = feature["dephead_ids"] + [0] * padding_word_len
                deplabel_ids = feature["deplabel_ids"] + [0] * padding_word_len
            else:
                dephead_ids = [0] * max_seq_len
                deplabel_ids = [0] * max_seq_len
            if self.parser_mode == PaserModeType.crf:
                label_ids = feature["label_ids"] + [0] * padding_word_len
                batch[i] = {"input_ids": input_ids,"attention_mask": mask,
                            "token_type_ids": type_ids,"orig_to_tok_index": orig_to_tok_index,
                            "word_seq_len": feature["word_seq_len"],
                            "dephead_ids": np.asarray(dephead_ids), "deplabel_ids": np.asarray(deplabel_ids),
                            "label_ids": label_ids}
            else:
                labels = []
                all_span_ids = []
                for x in feature["spanlabel_ids"]:                     # pading span labels
                    m1 = x[0]
                    label = x[1]
                    all_span_ids.append((m1[0], m1[1]))
                    labels.append(label)
                padding_span_len = max_span_num - len(labels)
                labels += [-1] * padding_span_len
                all_span_ids += [(0, 0)] * padding_span_len

                all_span_weight = feature["span_weight"] + [0] * padding_span_len
                all_span_lens = feature["span_lens"] + [0] * padding_span_len
                all_span_mask = feature["span_mask"] + [0] * padding_span_len

                batch[i] = {"input_ids": input_ids,
                            "attention_mask": mask,
                            "token_type_ids": type_ids,
                            "orig_to_tok_index": orig_to_tok_index,
                            "word_seq_len": feature["word_seq_len"],
                            "dephead_ids": dephead_ids, "deplabel_ids": deplabel_ids,
                            "all_span_ids": all_span_ids, "all_span_weight": all_span_weight,
                            "all_span_lens": all_span_lens, "all_span_mask": all_span_mask, "label_ids": labels}
        encoded_inputs = {key: [example[key] for example in batch] for key in batch[0].keys()}
        results = BatchEncoding(encoded_inputs, tensor_type='pt')
        return results


## testing code to test the dataset
if __name__ == '__main__':
    from transformers import RobertaTokenizerFast
    # from transformers import DebertaTokenizer
    # from transformers import RobertaTokenizer
    # tokenizer = RobertaTokenizerFast.from_pretrained('../../roberta-base', add_prefix_space=True)
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', add_prefix_space=True)
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-electra-base-discriminator',add_prefix_space=True)
    dataset = TransformersNERDataset(parser_mode=PaserModeType.span, dep_model=DepModelType.none, file="../../data/chinese/test.txt",tokenizer=tokenizer, is_train=True)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=dataset.collate_to_max_length)
    print(len(train_dataloader))
    for batch in train_dataloader:
        # print(batch.input_ids.size())
        print(batch.input_ids)
        pass
