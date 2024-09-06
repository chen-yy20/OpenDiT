import torch.distributed
from videosys import CogVideoXConfig, VideoSysEngine
import os
import time
import torch.cuda

def run_base():
    # models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
    # change num_gpus for multi-gpu inference
    model_id = os.getenv("MODEL_ID")
    num_gpus = int(os.getenv("WORLD_SIZE"))
    config = CogVideoXConfig(model_id, num_gpus)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # num frames should be <= 49. resolution is fixed to 720p.
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    video = engine.generate(
        prompt=prompt,
        guidance_scale=6,
        num_inference_steps=50,
        num_frames=49,
    ).video[0]
    end_time = time.time()
    total_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated()
    rank = torch.distributed.get_rank()
    print(f"Rank {rank}: time:{total_time}s, peak_memory:{peak_memory}MB", flush=True)
    save_name = f"cogvideox_'{prompt}'"
    engine.save_video(video, f"./outputs/{save_name}.mp4")


def run_pab():
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_low_mem():
    config = CogVideoXConfig("THUDM/CogVideoX-2b", cpu_offload=True, vae_tiling=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_pab()
    # run_low_mem()
