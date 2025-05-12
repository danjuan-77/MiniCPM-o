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
import os
import tempfile
import traceback


import torch
import librosa
import soundfile as sf
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Video libraries
from decord import VideoReader, cpu
import json
from tqdm import tqdm
from typing import List, Optional
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_audioclips,
    ImageClip,
    concatenate_videoclips,
)
tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"

# Constants for model loading
MODEL_PATH = '/share/nlp/tuwenming/models/openbmb/MiniCPM-o-2_6'
ATTN_IMPL = 'sdpa'  # or 'flash_attention_2'
TORCH_DTYPE = torch.bfloat16
MAX_FRAMES = 64  # adjust if OOM
pmp_avl_ans_format = "answer={'category1_id1': '[x_min, y_min, x_max, y_max]', 'category2_id2': '[x_min, y_min, x_max, y_max]'}"
avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']
prompt_avl = f"""
        ctaegories list: {avl_cls_list}
        (1) There may be multiple sounding instances, you can choose instance categories from the given categories list.
        (2) The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        (3) The bbox format is: [x_min, y_min, x_max, y_max], where x_min, y_min represent the coordinates of the top-left corner. 
        (4) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14].
        Do not explain, you must strictly follow the format: {pmp_avl_ans_format}
    """

prompt_avlg = """
        Please output the answer in a format that strictly matches the following example, do not explain:
        answer={'frame_0': [x0_min, y0_min, x0_max, y0_max], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}
        Note, 
        (1) x_min, y_min represent the coordinates of the top-left corner, while x_max, y_max for the bottom_right corner.
        (2) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14]. 
        (3) Frames should be ranged from frame_0 to frame_9.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },


    'L4_AVC': {

    },

    'L4_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L4_AVQA': {
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L5_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L5_AVQA': {
        'avqa_cls_list': avqa_cls_list,
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },
}

def concat_audio(audio_paths: List[str]) -> str:
    """
    Concatenate multiple audio files into one WAV file.
    Returns the path to the temp WAV file.
    """
    clips = [AudioFileClip(p) for p in audio_paths]
    final = concatenate_audioclips(clips)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    final.write_audiofile(out_path, fps=16000, logger=None)
    return out_path

def images_to_video(image_paths: List[str], duration: float, fps: int = 1) -> str:
    """
    Turn a list of images into a silent video of total `duration` seconds.
    Each image is shown for `duration / len(image_paths)` seconds.
    Returns the path to the temp MP4 file.
    """
    single_dur = duration / len(image_paths)
    clips = [ImageClip(p).set_duration(single_dur) for p in image_paths]
    video = concatenate_videoclips(clips, method="compose")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    video.write_videofile(out_path, fps=fps, codec="libx264", audio=False, logger=None)
    return out_path

def images_and_audio_to_video(image_paths: List[str], audio_paths: List[str], fps: int = 1) -> str:
    """
    Concatenate audio_paths into one audio, then build a video from image_paths
    that matches the audio duration, and merge them.
    Returns the path to the temp MP4 file.
    """
    # 1) build the concatenated audio
    audio_path = concat_audio(audio_paths)
    audio_clip = AudioFileClip(audio_path)
    # 2) build video from images matching audio duration
    duration = audio_clip.duration
    vid_path = images_to_video(image_paths, duration, fps=fps)
    # 3) attach audio to video
    video_clip = AudioFileClip(audio_path)  # re-open to avoid MoviePy caching issues
    from moviepy.editor import VideoFileClip
    base_vid = VideoFileClip(vid_path)
    final = base_vid.set_audio(audio_clip)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    final.write_videofile(out_path, fps=fps, codec="libx264", logger=None)
    return out_path 
    
def get_real_path(task_path: str, src_path: str) -> str:
    """传入taskpath和一些文件的path，构造文件的真实path

    Args:
        task_path (str): task path
        src_path (str): 每个文件的path

    Returns:
        str: 文件的真实path
    """
    temp_path = os.path.join(task_path, src_path)
    return os.path.normpath(temp_path)

def get_real_options_or_classes(d: dict) -> str:
    """Replace pseudo-options with real options text."""
    opts = d['input']['question'].get('options')
    if opts in havib_constants.get(d['task'], {}):
        opts = havib_constants[d['task']][opts]
    if opts:
        label = 'semantic categories' if 'cls' in opts else 'options'
        return f"Available {label} are: {opts}"
    return ''

def get_real_prompt(d: dict) -> str:
    """Replace pseudo-prompt with real prompt text."""
    prm = d['input']['question'].get('prompt')
    if prm in havib_constants.get(d['task'], {}):
        prm = havib_constants[d['task']][prm]
    return prm or ''

def get_real_input(d: dict) -> str:
    """Concatenate prompt, options, and question text into one input string."""
    prompt = get_real_prompt(d)
    options = get_real_options_or_classes(d)
    question = d['input']['question']['text'] or ''
    # 去掉多余的句点
    parts = [p for p in (prompt, options, question) if p]
    return " ".join(parts)

######################## Above are help function tools


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


def infer_text_audio(model, tokenizer, audio_path, text):
    """
    Inference for Text + Audio mode: transcribe or understand audio.
    """
    model.init_tts()
    # Load and preprocess audio
    audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
    prompt = text
    msgs = [{'role': 'user', 'content': [prompt, audio_input]}]

    # Run model.chat with TTS and generated audio
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        max_new_tokens=128,
        use_tts_template=True,
        generate_audio=False,
        temperature=0.3,
        # output_audio_path=output_audio
    )
    return answer


def infer_text_image(model, tokenizer, image_path, text):
    """
    Inference for Text + Image mode: describe static image.
    """
    image = Image.open(image_path).convert('RGB')
    question = text
    msgs = [{'role': 'user', 'content': [image, question]}]

    answer = model.chat(msgs=msgs, tokenizer=tokenizer)
    return answer


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


def infer_text_video(model, tokenizer, video_path, text):
    """
    Inference for Text + Video mode: describe video using sampled frames.
    """
    frames = encode_video_frames(video_path)
    question = text
    msgs = [{'role': 'user', 'content': frames + [question]}]

    # Additional decoding parameters
    params = {
        'use_image_id': False,
        'max_slice_nums': 2  # reduce if OOM
    }
    answer = model.chat(msgs=msgs, tokenizer=tokenizer, **params)
    return answer


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


def infer_video_with_audio(model, tokenizer, video_path, text):
    """
    Inference for Text + Video + Audio mode (omni):
    describe video content with audio context and TTS output.
    """
    model.init_tts()
    sys_msg = model.get_sys_prompt(mode='omni', language='en')
    chunks = get_video_chunk_content(video_path)

    # Build message sequence
    instruction = {'role': 'user', 'content': f'{text}'}
    msg = {'role': 'user', 'content': chunks}
    msgs = [sys_msg, instruction, msg]

    answer = model.chat(
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
        return_dict=False
    )
    return answer


def main():
    parser = argparse.ArgumentParser(description='Unified multimodal inference')
    # parser.add_argument('--mode', required=True, choices=[
    #     'text_audio', 'text_image', 'text_video', 'video_with_audio'
    # ], help='Inference mode to run')
    # parser.add_argument('--audio_path', help='Path to input audio file')
    # parser.add_argument('--image_path', help='Path to input image file')
    # parser.add_argument('--video_path', help='Path to input video file')
    parser.add_argument("--task_path", type=str, required=True, help="Path to the task folder containing data.json and media files")

    args = parser.parse_args()
    model, tokenizer = init_model()

    task_path = args.task_path
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    model_name = "minicpm-o"
    save_prediction_json = f'/share/nlp/tuwenming/projects/HAVIB/eval/user_outputs/{model_name}/tasks/{task_name}.json'
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)
    print('>>> save res to:', save_prediction_json)

    data_json_path = os.path.join(task_path, "data.json")
    with open(data_json_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    print(">>>Finished load raw data...")
    parsed_data = []
    for item in raw_data:
        inp = item.get('input', {})
        entry = {
            'id': item.get('id'),
            'task': item.get('task'),
            'subtask': item.get('subtask', None),
            'text': get_real_input(item),
            'audio_list': inp.get('audio_list', None),
            'image_list': inp.get('image_list', None),
            'video': inp.get('video', None)
        }
        parsed_data.append(entry)

    print(">>>Finished parse raw data...")    
    
    predictions = []
    
    for data in tqdm(parsed_data):
        _id = data['id']
        _task = data['task']
        _subtask = data['subtask']
        text = data['text']
        audio_list = (
            [get_real_path(task_path, p) for p in data["audio_list"]]
            if data["audio_list"] else None
        )
        image_list = (
            [get_real_path(task_path, p) for p in data["image_list"]]
            if data["image_list"] else None
        )
        video = (
            get_real_path(task_path, data['video'])
            if data['video'] else None
        )
        print(f">>> text input=:{text}")
        
        if audio_list and not image_list and not video:
            # Case 1: 仅音频
            audio_path = concat_audio(audio_list) if len(audio_list) > 1 else audio_list[0]
            output = infer_text_audio(model=model, tokenizer=tokenizer, audio_path=audio_path, text=text)
        
        elif image_list and not audio_list and not video:
            # Case 2: 仅图像
            image_path = image_list[0]
            output = infer_text_image(model=model, tokenizer=tokenizer, image_path=image_path, text=text)
        
        elif video and not audio_list and not image_list:
            # Case 3: 仅视频
            video_path = video
            output = infer_text_video(model=model, tokenizer=tokenizer, video_path=video_path, text=text)
        
        elif video and audio_list:
            # Case 4: 视频+音频
            audio_path = audio_list[0]
            if not os.path.exists(audio_path): # 去除audio
                video_path = video
                output = infer_text_video(model=model, tokenizer=tokenizer, video_path=video_path, text=text)
            elif not os.path.exists(video): # 去掉视频画面
                audio_path = concat_audio(audio_list) if len(audio_list) > 1 else audio_list[0]
                output = infer_text_audio(model=model, tokenizer=tokenizer, audio_path=audio_path, text=text)
            else:
                video_path = video
                output = infer_video_with_audio(model=model, tokenizer=tokenizer, video_path=video_path, text=text)

        elif image_list and audio_list and not video:
            # Case 5: 图像+音频 -> 合成视频, 使用视频的audio
            audio_path = audio_list[0]
            if not os.path.exists(audio_path):
                video_path = images_to_video(image_list, len(image_list), fps=1)
                output = infer_text_video(model=model, tokenizer=tokenizer, video_path=video_path, text=text)
            else:
                video_path = images_and_audio_to_video(image_list, audio_list, fps=1)
                output = infer_video_with_audio(model=model, tokenizer=tokenizer, video_path=video_path, text=text)
        
        else:
            raise ValueError(f"Unsupported input combination for id={_id}")

        pred_record = {
            "task": _task,
            "subtask": _subtask,
            "id": _id,
            "predict": output,
        }
        predictions.append(pred_record)
        print('>>> ans=:', pred_record)
    
    
    with open(save_prediction_json, 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    main()
