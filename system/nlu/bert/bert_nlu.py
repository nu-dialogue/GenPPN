from logging import getLogger
import os
import torch
import json
import zipfile
import spacy
from spacy.symbols import ORTH, LEMMA, POS
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.util.file_util import cached_path, get_root_path
from convlab2.nlu.jointBERT.dataloader import Dataloader
from convlab2.nlu.jointBERT.jointBERT import JointBERT
from convlab2.nlu.jointBERT.multiwoz.preprocess import preprocess

from utils import get_device, set_logger

logger = getLogger(__name__)
set_logger(logger)

class WrappedBERTNLU(BERTNLU):
    def __init__(self, mode="usr", config_file="full-usr.json", model_file=None) -> None:
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        self.mode = mode
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/{}'.format(config_file))
        config = json.load(open(config_file))
        # print(config['DEVICE'])
        # DEVICE = config['DEVICE']
        DEVICE = str(get_device()) # Just modified this line from cuda:0
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))

        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        print('Load from', best_model_path)
        model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.device = DEVICE
        model.eval()

        self.model = model
        self.use_context = config['model']['context']
        self.dataloader = dataloader
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            print('download en_core_web_sm for spacy')
            from spacy.cli.download import download as spacy_download
            spacy_download("en_core_web_sm")
            spacy_model_module = __import__("en_core_web_sm")
            self.nlp = spacy_model_module.load()
        with open(os.path.join(get_root_path(), 'data/multiwoz/db/postcode.json'), 'r') as f:
            token_list = json.load(f)

        for token in token_list:
            token = token.strip()
            self.nlp.tokenizer.add_special_case(token, [{ORTH: token, LEMMA: token, POS: u'NOUN'}])
        print("BERTNLU loaded")