#!/usr/bin/env python3
"""
Unified inference script supporting multiple multimodal modes:
  - Text + Audio
  - Text + Image
  - Text + Video (frames only)
  - Text + Video (frames + audio chunks)

Usage:
  python unified_inference.py --mode MODE [--audio_path PATH] [--image_path PATH] [--video_path PATH] [--output_audio PATH]

Example:
  python unified_inference.py --mode text_audio --audio_path ./assets/input_examples/audio_understanding.mp3
  python unified_inference.py --mode text_image --image_path ./assets/minicpmo2_6/show_demo.jpg
  python unified_inference.py --mode text_video --video_path ./examples/videos/needle_32.mp4
  python unified_inference.py --mode video_with_audio --video_path ./examples/videos/Skiing.mp4 --output_audio result.wav
"""
import argparse
import math
import tempfile

import torch
import librosa
import soundfile as sf
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Video libraries
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip

# Constants for model loading
MODEL_PATH = '/share/nlp/tuwenming/models/openbmb/MiniCPM-o-2_6'
ATTN_IMPL = 'sdpa'  # or 'flash_attention_2'
TORCH_DTYPE = torch.bfloat16
MAX_FRAMES = 64  # adjust if OOM


def init_model():
    """
    Load and initialize the MiniCPM-o model and tokenizer once.
    """
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        torch_dtype=TORCH_DTYPE
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    return model, tokenizer


def infer_text_audio(model, tokenizer, audio_path):
    """
    Inference for Text + Audio mode: transcribe or understand audio.
    """
    model.init_tts()
    # Load and preprocess audio
    audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
    prompt = "Please listen to the audio snippet carefully and transcribe the content.\n"
    msgs = [{'role': 'user', 'content': [prompt, audio_input]}]

    # Run model.chat with TTS and generated audio
    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        max_new_tokens=128,
        use_tts_template=True,
        generate_audio=False,
        temperature=0.3,
        # output_audio_path=output_audio
    )
    print(res)


def infer_text_image(model, tokenizer, image_path):
    """
    Inference for Text + Image mode: describe static image.
    """
    image = Image.open(image_path).convert('RGB')
    question = "What is the landform in the picture?"
    msgs = [{'role': 'user', 'content': [image, question]}]

    answer = model.chat(msgs=msgs, tokenizer=tokenizer)
    print(answer)

    # Example of a follow-up question
    msgs.append({'role': 'assistant', 'content': [answer]})
    msgs.append({'role': 'user', 'content': ['What should I pay attention to when traveling here?']})

    followup = model.chat(msgs=msgs, tokenizer=tokenizer)
    print(followup)


def encode_video_frames(video_path):
    """
    Sample and return a list of PIL images from the video frames.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    # Uniform sampling of frames if too many
    total = len(vr)
    sample_fps = round(vr.get_avg_fps())
    idxs = list(range(0, total, sample_fps))
    if len(idxs) > MAX_FRAMES:
        # reduce to MAX_FRAMES uniformly
        gap = len(idxs) / MAX_FRAMES
        idxs = [int(i * gap + gap / 2) for i in range(MAX_FRAMES)]

    frames = vr.get_batch(idxs).asnumpy()
    pil_frames = [Image.fromarray(f.astype('uint8')) for f in frames]
    print('Sampled frames:', len(pil_frames))
    return pil_frames


def infer_text_video(model, tokenizer, video_path):
    """
    Inference for Text + Video mode: describe video using sampled frames.
    """
    frames = encode_video_frames(video_path)
    question = "Describe the video"
    msgs = [{'role': 'user', 'content': frames + [question]}]

    # Additional decoding parameters
    params = {
        'use_image_id': False,
        'max_slice_nums': 2  # reduce if OOM
    }
    answer = model.chat(msgs=msgs, tokenizer=tokenizer, **params)
    print(answer)


def get_video_chunk_content(video_path, flatten=True):
    """
    Split video into 1-second chunks of frame + audio.
    Returns a flat or nested list of ['<unit>', image, audio_array].
    """
    clip = VideoFileClip(video_path)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
        clip.audio.write_audiofile(tmp.name, codec='pcm_s16le', fps=16000)
        audio_np, sr = librosa.load(tmp.name, sr=16000, mono=True)
    duration = math.ceil(clip.duration)

    contents = []
    for i in range(duration):
        frame = clip.get_frame(i + 1)
        img = Image.fromarray(frame.astype('uint8'))
        auditory = audio_np[sr * i:sr * (i + 1)]
        segment = ['<unit>', img, auditory]
        contents.extend(segment if flatten else [segment])
    return contents


def infer_video_with_audio(model, tokenizer, video_path):
    """
    Inference for Text + Video + Audio mode (omni):
    describe video content with audio context and TTS output.
    """
    model.init_tts()
    sys_msg = model.get_sys_prompt(mode='omni', language='en')
    chunks = get_video_chunk_content(video_path)

    # Build message sequence
    instruction = {'role': 'user', 'content': 'describe the video'}
    msg = {'role': 'user', 'content': chunks}
    msgs = [instruction, sys_msg, msg]

    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.5,
        max_new_tokens=4096,
        omni_input=True,
        use_tts_template=True,
        generate_audio=False,
        # output_audio_path=output_audio,
        max_slice_nums=1,
        use_image_id=False,
        return_dict=True
    )
    print(res)


def main():
    parser = argparse.ArgumentParser(description='Unified multimodal inference')
    parser.add_argument('--mode', required=True, choices=[
        'text_audio', 'text_image', 'text_video', 'video_with_audio'
    ], help='Inference mode to run')
    parser.add_argument('--audio_path', help='Path to input audio file')
    parser.add_argument('--image_path', help='Path to input image file')
    parser.add_argument('--video_path', help='Path to input video file')

    args = parser.parse_args()
    model, tokenizer = init_model()

    if args.mode == 'text_audio':
        infer_text_audio(model, tokenizer, args.audio_path)
    elif args.mode == 'text_image':
        infer_text_image(model, tokenizer, args.image_path)
    elif args.mode == 'text_video':
        infer_text_video(model, tokenizer, args.video_path)
    elif args.mode == 'video_with_audio':
        infer_video_with_audio(model, tokenizer, args.video_path)
    else:
        parser.error('Unknown mode: ' + args.mode)


if __name__ == '__main__':
    main()
