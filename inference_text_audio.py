import torch
import librosa
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('/share/nlp/tuwenming/models/openbmb/MiniCPM-o-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('/share/nlp/tuwenming/models/openbmb/MiniCPM-o-2_6', trust_remote_code=True)

model.init_tts()
model.tts.float()


task_prompt = "Please listen to the audio snippet carefully and transcribe the content." + "\n" # can change to other prompts.
audio_input, _ = librosa.load('./assets/input_examples/audio_understanding.mp3', sr=16000, mono=True) # load the audio to be captioned

msgs = [{'role': 'user', 'content': [task_prompt, audio_input]}]

res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result_audio_understanding.wav',
)
print(res)
