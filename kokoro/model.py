from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from .voice_probe import emit_probe, tensor_stats
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None
        duration: Optional[torch.FloatTensor] = None  # continuous pre-round duration, shape [1, n_phonemes]

    def _forward_with_tokens_impl(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        probe_id: Optional[str] = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        emit_probe(
            "model.forward_with_tokens.pre",
            probe_id=probe_id,
            input_ids=tensor_stats(input_ids),
            ref_s=tensor_stats(ref_s),
            speed=float(speed),
        )
        input_lengths = torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[-1], 
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        # During training, cap total predicted frames so the alignment matrix and iSTFT
        # never grow unboundedly large.  LJSpeech is capped at 10 s = ~937 frames; allow
        # up to 1.5× that (1400 frames ≈ 14.9 s) before rescaling.  The continuous
        # `duration` tensor is left untouched so dur_loss still pulls it toward the target.
        if torch.is_grad_enabled():
            total_frames = pred_dur.sum()
            _MAX_TRAIN_FRAMES = 1400
            if total_frames > _MAX_TRAIN_FRAMES:
                pred_dur = (pred_dur.float() * (_MAX_TRAIN_FRAMES / total_frames.float())).round().clamp(min=1).long()
        emit_probe(
            "model.forward_with_tokens.duration",
            probe_id=probe_id,
            raw_duration=tensor_stats(duration),
            pred_dur=tensor_stats(pred_dur),
        )
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        emit_probe(
            "model.forward_with_tokens.prosody",
            probe_id=probe_id,
            alignment=tensor_stats(pred_aln_trg),
            F0_pred=tensor_stats(F0_pred),
            N_pred=tensor_stats(N_pred),
        )
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        emit_probe(
            "model.forward_with_tokens.post",
            probe_id=probe_id,
            asr=tensor_stats(asr),
            audio=tensor_stats(audio),
        )
        return audio, pred_dur, duration

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        probe_id: Optional[str] = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        audio, pred_dur, _ = self._forward_with_tokens_impl(
            input_ids=input_ids, ref_s=ref_s, speed=speed, probe_id=probe_id
        )
        return audio, pred_dur

    def forward_with_tokens_trainable(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        probe_id: Optional[str] = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        return self._forward_with_tokens_impl(
            input_ids=input_ids, ref_s=ref_s, speed=speed, probe_id=probe_id
        )

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
        probe_id: Optional[str] = None,
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        emit_probe(
            "model.forward.pre",
            probe_id=probe_id,
            phoneme_len=len(phonemes),
            token_count=len(input_ids),
        )
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed, probe_id=probe_id)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        emit_probe(
            "model.forward.post",
            probe_id=probe_id,
            audio=tensor_stats(audio),
            pred_dur=tensor_stats(pred_dur) if pred_dur is not None else None,
        )
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

    def forward_trainable(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
        probe_id: Optional[str] = None,
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        emit_probe(
            "model.forward_trainable.pre",
            probe_id=probe_id,
            phoneme_len=len(phonemes),
            token_count=len(input_ids),
        )
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur, duration = self.forward_with_tokens_trainable(input_ids, ref_s, speed, probe_id=probe_id)
        audio = audio.squeeze()
        emit_probe(
            "model.forward_trainable.post",
            probe_id=probe_id,
            audio=tensor_stats(audio.detach().cpu()),
            pred_dur=tensor_stats(pred_dur.detach().cpu()) if pred_dur is not None else None,
        )
        if return_output:
            pred_dur_out = pred_dur if pred_dur is not None else None
            return self.Output(audio=audio, pred_dur=pred_dur_out, duration=duration)
        return audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
