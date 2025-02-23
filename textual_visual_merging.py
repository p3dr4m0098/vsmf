import os
import json
import shutil
import requests
import threading
import subprocess
from pathlib import Path
from textual_merger import NLP_merging
from visual_merger import visual_merging
from vision_summarizer import vision_summarizer
from moviepy.editor import VideoFileClip, concatenate_videoclips
from final_textual_visual_merger import find_overlapping_intervals

semaphore = threading.Semaphore(0)

def clipping_and_merging(input_path, intervals, output_path):
    trim_args = ""
    for start, end in intervals:
        trim_args += f"{start},{end},"
    trim_args = trim_args[:-1]
    command = ["./smartcut_linux", input_path, output_path, "--keep", trim_args]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Command executed successfully!")
        print(result.stdout)  
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return -1


def extract_audio(input_mp4_path, output_audio_path):
    video = VideoFileClip(input_mp4_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path, codec='mp3')


def vision_thread(first_video_dir, first_video_name):
    global visual_ints, max_du
    vision_summarizer(first_video_dir, first_video_name)
    visual_ints, max_du = visual_merging(first_video_dir)
    semaphore.release()

def asr_thread(mp3_file, first_video_dir, first_video_name):
    global asr_ints
    
    headers = {'Authorization': 'Token 87f6196c6f7fa68f3c14156cc69a09631bf35f8f',}
    files = {'media': (mp3_file, open(mp3_file, 'rb')),}
    response = requests.post('https://harf.roshan-ai.ir/api/transcribe_files/', headers=headers, files=files)


    with open(os.path.join(first_video_dir, f"{first_video_name}_ASR.json"), 'w', encoding='utf-8') as f:  
        json.dump(json.loads(response.text), f, ensure_ascii=False, indent=4)  

    with open(os.path.join(first_video_dir, f"{first_video_name}_ASR.json"), 'r', encoding='utf-8') as f_new:
        asr_data = json.load(f_new)

    asr_ints = NLP_merging(asr_data, os.path.join(first_video_dir, f"{first_video_name}.mp4"))

    
    semaphore.release()

def merger(input_video_path, output_video_path):

    first_video_name = Path(input_video_path).stem
    first_video_dir = f"./videos/{first_video_name}"
    
    os.makedirs(first_video_dir, exist_ok=True)
    shutil.copy(input_video_path, os.path.join(first_video_dir, f"{first_video_name}.mp4"))
    extract_audio(os.path.join(first_video_dir, f"{first_video_name}.mp4"), \
                 os.path.join(first_video_dir, f"{first_video_name}.mp3"))

    t1 = threading.Thread(target=vision_thread, args=(first_video_dir, first_video_name, ))
    t2 = threading.Thread(target=asr_thread, args=(os.path.join(first_video_dir, f"{first_video_name}.mp3"), first_video_dir, first_video_name, ))
    
    t1.start()
    t2.start()

    semaphore.acquire()  
    semaphore.acquire()  

    final_overlaps = find_overlapping_intervals(asr_ints, visual_ints, max_du)

    clipping_and_merging(os.path.join(first_video_dir, f"{first_video_name}.mp4"), \
                         final_overlaps, output_video_path)
                         
merger("sh_iran_english.mp4", "output_sh_iran_english.mp4")
