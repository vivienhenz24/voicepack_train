#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_audio_mono_24k(path: str) -> np.ndarray:
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
    except Exception:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            sampwidth = wf.getsampwidth()
            ch = wf.getnchannels()
        if sampwidth == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            raise RuntimeError("Only 16-bit PCM WAV fallback is supported without soundfile")
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        audio = x

    if sr != 24000:
        t_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_len = int(round(len(audio) * (24000.0 / sr)))
        t_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(t_new, t_old, audio).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def write_wav16(path: str, audio: np.ndarray, sr: int = 24000) -> None:
    x = np.asarray(audio, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def stft_mag(x: torch.Tensor, n_fft: int = 1024, hop: int = 256, win: int = 1024) -> torch.Tensor:
    w = torch.hann_window(win, device=x.device)
    z = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
    return z.abs()


def parse_manifest(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("target_audio") or not row.get("text"):
                continue
            out.append(row)
    return out


def split_rows(rows: Sequence[Dict[str, str]], val_ratio: float, seed: int) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    items = list(rows)
    rng = random.Random(seed)
    rng.shuffle(items)
    if val_ratio <= 0:
        return items, []
    n_val = max(1, int(round(len(items) * val_ratio)))
    n_val = min(n_val, max(0, len(items) - 1))
    return items[n_val:], items[:n_val]


@dataclass
class Sample:
    sample_id: str
    target_audio: str
    text: str
    phonemes: str
    pack_index: int


def build_samples(pipe, rows: Sequence[Dict[str, str]]) -> List[Sample]:
    samples: List[Sample] = []
    for i, row in enumerate(rows):
        text = row["text"]
        sample_id = (row.get("id") or f"sample_{i:06d}").strip()
        if pipe.lang_code in "ab":
            _, tokens = pipe.g2p(text)
            ps = ""
            for _gs, part_ps, _ in pipe.en_tokenize(tokens):
                if part_ps:
                    ps = part_ps
                    break
        else:
            ps, _ = pipe.g2p(text)
        if not ps:
            continue
        idx = max(0, min(len(ps) - 1, 509))
        samples.append(
            Sample(
                sample_id=sample_id,
                target_audio=row["target_audio"],
                text=text,
                phonemes=ps,
                pack_index=idx,
            )
        )
    return samples


def smoothness_reg(pack: torch.Tensor) -> torch.Tensor:
    # Penalize curvature across length axis to keep row trajectory smooth.
    d2 = pack[2:] - 2.0 * pack[1:-1] + pack[:-2]
    return d2.pow(2).mean()


def split_norm_reg(pack: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    p = pack.squeeze(1)
    b = base.squeeze(1)
    p_l = torch.norm(p[:, :128], dim=-1)
    p_r = torch.norm(p[:, 128:], dim=-1)
    b_l = torch.norm(b[:, :128], dim=-1)
    b_r = torch.norm(b[:, 128:], dim=-1)
    return F.l1_loss(p_l, b_l) + F.l1_loss(p_r, b_r)


def eval_dataset(
    pipe,
    pack_param: torch.Tensor,
    samples: Sequence[Sample],
    max_audio_seconds: float,
    speed: float,
) -> Dict[str, float]:
    if not samples:
        return {"val_audio_l1": 0.0, "val_spec_l1": 0.0}
    total_audio = 0.0
    total_spec = 0.0
    with torch.no_grad():
        for s in samples:
            target_np = read_audio_mono_24k(s.target_audio)
            cap = int(24000 * max_audio_seconds)
            target = torch.from_numpy(target_np[:cap]).to(pipe.model.device)
            ref = pack_param[s.pack_index]
            out = pipe.model(s.phonemes, ref, speed=speed, return_output=True)
            pred = out.audio
            n = min(pred.numel(), target.numel())
            pred_c = pred[:n]
            tgt_c = target[:n]
            total_audio += float(F.l1_loss(pred_c, tgt_c).detach().cpu())
            total_spec += float(F.l1_loss(torch.log1p(stft_mag(pred_c)), torch.log1p(stft_mag(tgt_c))).detach().cpu())
    denom = float(len(samples))
    return {"val_audio_l1": total_audio / denom, "val_spec_l1": total_spec / denom}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a full Kokoro voice pack from a manifest (reconstructed loop)")
    ap.add_argument("--manifest-csv", required=True, help="CSV with id,target_audio,text and optional lang/voice_init")
    ap.add_argument("--lang", default="a")
    ap.add_argument("--voice-init", default="af_heart")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-audio-seconds", type=float, default=20.0)
    ap.add_argument("--max-grad-phonemes", type=int, default=None,
                    help="Truncate phoneme string to this length before forward_trainable. "
                         "Controls output audio length and thus activation memory. "
                         "pack_index is preserved from the full string.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None, help="cpu|cuda|mps")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--w-wav", type=float, default=1.0)
    ap.add_argument("--w-spec", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=0.05, help="L2 anchor to init pack")
    ap.add_argument("--w-smooth", type=float, default=0.02, help="2nd-difference smoothness across rows")
    ap.add_argument("--w-splitnorm", type=float, default=0.02, help="Preserve left/right row-norm profile")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--out-dir", default="logs/voice_loop")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from kokoro import KPipeline

    pipe = KPipeline(lang_code=args.lang, model=True, device=args.device)
    for p in pipe.model.parameters():
        p.requires_grad = False
    # KPipeline sets eval() internally; cuDNN RNN backward requires train mode.
    # Params are frozen so train() here won't cause weight updates.
    pipe.model.train()

    raw_rows = parse_manifest(args.manifest_csv)
    if not raw_rows:
        raise SystemExit("Manifest has no valid rows")

    train_rows, val_rows = split_rows(raw_rows, args.val_ratio, args.seed)
    train_samples = build_samples(pipe, train_rows)
    val_samples = build_samples(pipe, val_rows)
    if not train_samples:
        raise SystemExit("No valid train samples after phonemization")

    base_pack = pipe.load_voice(args.voice_init).to(pipe.model.device).detach()
    if tuple(base_pack.shape) != (510, 1, 256):
        raise SystemExit(f"Unexpected voice pack shape: {tuple(base_pack.shape)}")

    pack_param = torch.nn.Parameter(base_pack.clone())
    opt = torch.optim.Adam([pack_param], lr=args.lr)

    history_path = out_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    print(f"train_samples={len(train_samples)} val_samples={len(val_samples)}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_samples)))
        random.shuffle(order)
        epoch_losses: Dict[str, float] = {"loss": 0.0, "wav": 0.0, "spec": 0.0, "anchor": 0.0, "smooth": 0.0, "splitnorm": 0.0}

        for pos, i in enumerate(order, start=1):
            s = train_samples[i]

            _cuda = torch.cuda.is_available()
            if _cuda and pos % 50 == 1:
                _alloc = torch.cuda.memory_allocated() / 1024**3
                _reserv = torch.cuda.memory_reserved() / 1024**3
                print(f"  [mem] step={pos} alloc={_alloc:.2f}GB reserved={_reserv:.2f}GB", flush=True)

            target_np = read_audio_mono_24k(s.target_audio)
            cap = int(24000 * args.max_audio_seconds)
            target = torch.from_numpy(target_np[:cap]).to(pipe.model.device)

            if _cuda and pos % 50 == 1:
                print(f"  [mem] after target load: alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB  audio_samples={target.numel()}  phoneme_len={len(s.phonemes)}", flush=True)

            opt.zero_grad(set_to_none=True)
            ref = pack_param[s.pack_index]
            out = pipe.model.forward_trainable(s.phonemes, ref, speed=args.speed, return_output=True)
            pred = out.audio

            if _cuda and pos % 50 == 1:
                print(f"  [mem] after forward: alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB  pred_samples={pred.numel()}", flush=True)

            n = min(pred.numel(), target.numel())
            pred_c = pred[:n]
            tgt_c = target[:n]

            wav_l1 = F.l1_loss(pred_c, tgt_c)
            spec_l1 = F.l1_loss(torch.log1p(stft_mag(pred_c)), torch.log1p(stft_mag(tgt_c)))
            anchor = (pack_param - base_pack).pow(2).mean()
            smooth = smoothness_reg(pack_param)
            splitn = split_norm_reg(pack_param, base_pack)

            loss = (
                args.w_wav * wav_l1
                + args.w_spec * spec_l1
                + args.w_anchor * anchor
                + args.w_smooth * smooth
                + args.w_splitnorm * splitn
            )
            loss.backward()

            if _cuda and pos % 50 == 1:
                print(f"  [mem] after backward: alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([pack_param], args.clip_grad)
            opt.step()

            if _cuda and pos % 50 == 1:
                print(f"  [mem] after step: alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

            row = {
                "epoch": epoch,
                "step": global_step,
                "sample_id": s.sample_id,
                "pack_index": s.pack_index,
                "loss": float(loss.detach().cpu()),
                "wav_l1": float(wav_l1.detach().cpu()),
                "spec_l1": float(spec_l1.detach().cpu()),
                "anchor": float(anchor.detach().cpu()),
                "smooth": float(smooth.detach().cpu()),
                "splitnorm": float(splitn.detach().cpu()),
            }
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True))
                f.write("\n")

            epoch_losses["loss"] += row["loss"]
            epoch_losses["wav"] += row["wav_l1"]
            epoch_losses["spec"] += row["spec_l1"]
            epoch_losses["anchor"] += row["anchor"]
            epoch_losses["smooth"] += row["smooth"]
            epoch_losses["splitnorm"] += row["splitnorm"]

            global_step += 1
            if pos % args.log_every == 0 or pos == len(order):
                print(
                    f"epoch={epoch}/{args.epochs} sample={pos}/{len(order)} "
                    f"loss={row['loss']:.5f} wav={row['wav_l1']:.5f} spec={row['spec_l1']:.5f} "
                    f"idx={s.pack_index}"
                )

        denom = float(len(order))
        train_summary = {
            "train_loss": epoch_losses["loss"] / denom,
            "train_wav_l1": epoch_losses["wav"] / denom,
            "train_spec_l1": epoch_losses["spec"] / denom,
            "train_anchor": epoch_losses["anchor"] / denom,
            "train_smooth": epoch_losses["smooth"] / denom,
            "train_splitnorm": epoch_losses["splitnorm"] / denom,
        }
        val_summary = eval_dataset(
            pipe=pipe,
            pack_param=pack_param.detach(),
            samples=val_samples,
            max_audio_seconds=args.max_audio_seconds,
            speed=args.speed,
        )
        summary = {"epoch": epoch, **train_summary, **val_summary}
        print(json.dumps(summary, ensure_ascii=True))

        if args.save_every_epoch:
            torch.save(pack_param.detach().cpu(), out_dir / f"voice_pack_epoch{epoch:03d}.pt")

    final_pack = pack_param.detach().cpu()
    torch.save(final_pack, out_dir / "voice_pack_trained.pt")
    torch.save(base_pack.detach().cpu(), out_dir / "voice_pack_init.pt")

    # Quick sanity synthesis using first train sample.
    demo = train_samples[0]
    with torch.no_grad():
        out = pipe.model(demo.phonemes, pack_param.detach()[demo.pack_index], speed=args.speed, return_output=True)
    write_wav16(str(out_dir / "demo_train_sample.wav"), out.audio.detach().cpu().numpy())

    meta = {
        "manifest_csv": args.manifest_csv,
        "lang": args.lang,
        "voice_init": args.voice_init,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "weights": {
            "w_wav": args.w_wav,
            "w_spec": args.w_spec,
            "w_anchor": args.w_anchor,
            "w_smooth": args.w_smooth,
            "w_splitnorm": args.w_splitnorm,
        },
        "outputs": {
            "trained_pack": str(out_dir / "voice_pack_trained.pt"),
            "history": str(history_path),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"trained_pack={out_dir / 'voice_pack_trained.pt'}")
    print(f"history={history_path}")
    print(f"metadata={out_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
