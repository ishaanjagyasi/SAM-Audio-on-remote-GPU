import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor
import argparse

def separate_audio(audio_path, description, output_prefix="output", rerank=0):
    device = "cuda"
    dtype = torch.float16

    model = SAMAudio.from_pretrained("facebook/sam-audio-large")
    print("dtype before cuda:", next(model.parameters()).dtype)
    model = model.half().to("cuda").eval()
    print("dtype after cuda:", next(model.parameters()).dtype)

    processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")

    inputs = processor(audios=[audio_path], descriptions=[description])
    inputs = inputs.to(device)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        import soundfile as sf
        import numpy as np
        import os

        wav, sr_in = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        chunk_sec = 10
        chunk_len = int(chunk_sec * sr_in)
        tgt_chunks = []
        res_chunks = []

        for start in range(0, len(wav), chunk_len):
            end = min(start + chunk_len, len(wav))
            chunk = wav[start:end]
            tmp_path = f"{output_prefix}__tmp_chunk.wav"
            sf.write(tmp_path, chunk, sr_in)

            chunk_inputs = processor(audios=[tmp_path], descriptions=[description]).to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                chunk_result = model.separate(chunk_inputs, predict_spans=False, reranking_candidates=rerank)

            tgt_chunks.append(chunk_result.target[0].detach().cpu().numpy())
            res_chunks.append(chunk_result.residual[0].detach().cpu().numpy())

            torch.cuda.empty_cache()
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        target = np.concatenate(tgt_chunks)
        residual = np.concatenate(res_chunks)

        sf.write(f"{output_prefix}_target.wav", target.astype("float32"), processor.audio_sampling_rate)
        sf.write(f"{output_prefix}_residual.wav", residual.astype("float32"), processor.audio_sampling_rate)
        return

    sr = processor.audio_sampling_rate
    torchaudio.save(f"{output_prefix}_target.wav", result.target[0].unsqueeze(0).cpu(), sr)
    torchaudio.save(f"{output_prefix}_residual.wav", result.residual[0].unsqueeze(0).cpu(), sr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--desc", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rerank", type=int, default=0)
    args = p.parse_args()
    separate_audio(args.audio, args.desc, args.out, args.rerank)
