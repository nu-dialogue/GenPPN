import json
import os
import zipfile
import sys
from os.path import dirname
import random
import argparse 
from collections import Counter

CURRENT_DPATH = dirname(os.path.abspath(__file__))
ROOT_DPATH = dirname(dirname(dirname(CURRENT_DPATH)))
sys.path.append(ROOT_DPATH)

def read_zipped_json(filepath, filename):
    print(filepath)
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def da2triples(dialog_act):
    triples = []
    for intent, svs in dialog_act.items():
        for slot, value in svs:
            triples.append([intent, slot, value])
    return triples


def preprocess(mode, data_id):
    data_dir = os.path.join(ROOT_DPATH, 'ConvLab-2/data/multiwoz')
    processed_data_dir = os.path.join(CURRENT_DPATH, f'multiwoz_data/{data_id}')
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    keys = ['train', 'val', 'test']

    processed_data = {}
    all_da = []
    all_intent = []
    all_tag = []
    context_size = 3
    for key in keys:
        key_data = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        dialogs_to_use = {dial_name: [True]*len(dial["log"]) for dial_name, dial in key_data.items()}
        
        processed_data[key] = []
        for sess in dialogs_to_use:
            context = []
            for i, to_use in enumerate(dialogs_to_use[sess]):
                if not to_use:
                    continue
                turn = key_data[sess]["log"][i]
                if mode == 'usr' and i % 2 == 1:
                    context.append(turn['text'])
                    continue
                elif mode == 'sys' and i % 2 == 0:
                    context.append(turn['text'])
                    continue
                tokens = turn["text"].split()
                dialog_act = {}
                for dacts in turn["span_info"]:
                    if dacts[0] not in dialog_act:
                        dialog_act[dacts[0]] = []
                    dialog_act[dacts[0]].append([dacts[1], " ".join(tokens[dacts[3]: dacts[4] + 1])])

                spans = turn["span_info"]
                tags = []
                for i, _ in enumerate(tokens):
                    for span in spans:
                        if i == span[3]:
                            tags.append("B-" + span[0] + "+" + span[1])
                            break
                        if span[3] < i <= span[4]:
                            tags.append("I-" + span[0] + "+" + span[1])
                            break
                    else:
                        tags.append("O")

                intents = []
                for dacts in turn["dialog_act"]:
                    for dact in turn["dialog_act"][dacts]:
                        if dacts not in dialog_act or dact[0] not in [sv[0] for sv in dialog_act[dacts]]:
                            if dact[1] in ["none", "?", "yes", "no", "do nt care", "do n't care", "dontcare"]:
                                intents.append(dacts + "+" + dact[0] + "*" + dact[1])
                processed_data[key].append([tokens, tags, intents, da2triples(turn["dialog_act"]), context[-context_size:]])
                all_da += [da for da in turn['dialog_act']]
                all_intent += intents
                all_tag += tags

                context.append(turn['text'])

        all_da = [x[0] for x in dict(Counter(all_da)).items() if x[1]]
        all_intent = [x[0] for x in dict(Counter(all_intent)).items() if x[1]]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items() if x[1]]

        print('loaded {}, size {}'.format(key, len(processed_data[key])))
        json.dump(processed_data[key], open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w'), indent=2)

    print('dialog act num:', len(all_da))
    print('sentence label num:', len(all_intent))
    print('tag num:', len(all_tag))
    json.dump(all_da, open(os.path.join(processed_data_dir, 'all_act.json'), 'w'), indent=2)
    json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w'), indent=2)
    json.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.json'), 'w'), indent=2)

def make_training_config(data_id, use_context):
    config = {
        "data_dir": os.path.join(CURRENT_DPATH, f"multiwoz_data/{data_id}"),
        "output_dir": os.path.join(CURRENT_DPATH, f"outputs/{data_id}"),
        "log_dir": os.path.join(CURRENT_DPATH, f"log/{data_id}"),
        "zipped_model_path": os.path.join(CURRENT_DPATH, f"outputs/{data_id}/{data_id}.zip"),
        "DEVICE": "cuda:0",
        "seed": 42,
        "cut_sen_len": 40,
        "use_bert_tokenizer": True,
        "model": {
            "finetune": True,
            "context": use_context,
            "context_grad": False,
            "pretrained_weights": "bert-base-uncased",
            "check_step": 1000,
            "max_step": 10000,
            "batch_size": 40,
            "learning_rate": 1e-4,
            "adam_epsilon": 1e-8,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "dropout": 0.1,
            "hidden_units": 1536
        }
    }
    config_fpath = os.path.join(CURRENT_DPATH, f"configs/{data_id}.json")
    json.dump(config, open(config_fpath, 'w'), indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=str)
    parser.add_argument('--mode', type=str, default="usr",
                        help="Side of the target utterance to be trained. 'usr' or 'sys'")
    parser.add_argument('--use_context', action='store_true',
                        help="Whether to input context (last 3 turns) to the bert model or not")
    args = parser.parse_args()
    preprocess(data_id=args.data_id, mode=args.mode)
    make_training_config(data_id=args.data_id, use_context=args.use_context)
