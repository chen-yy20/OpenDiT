from videosys import LatteConfig, VideoSysEngine
import os
import time
import torch.cuda
import torch.distributed
import statistics

def run_base():
    model_id = os.getenv("MODEL_ID")
    num_gpus = int(os.getenv("WORLD_SIZE"))
    config = LatteConfig(model_id, num_gpus)
    engine = VideoSysEngine(config)

    prompts = [
        "Sunset over the sea.",
        "A bustling cityscape at night.",
        "A field of blooming sunflowers.",
        "A snow-capped mountain peak.",
        "A tranquil forest stream.",
    ]

    num_runs = len(prompts)
    times = []
    peak_memories = []

    rank = torch.distributed.get_rank()

    for i, prompt in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        video = engine.generate(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
        ).video[0]
        
        end_time = time.time()
        total_time = end_time - start_time
        peak_memory = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
        
        times.append(total_time)
        peak_memories.append(peak_memory)
        
        print(f"Rank {rank}, Run {i+1}/{num_runs}: Prompt: '{prompt}', Time: {total_time:.2f}s, Peak Memory: {peak_memory:.2f}GB", flush=True)
        
        save_name = f"latte_'{prompt}'"
        engine.save_video(video, f"./outputs/{save_name}.mp4")

    avg_time = statistics.mean(times)
    avg_memory = statistics.mean(peak_memories)
    
    print(f"Rank {rank}: Average Time: {avg_time:.2f}s, Average Peak Memory: {avg_memory:.2f}GB", flush=True)
    
    if rank == 0:
        print("\nAll runs completed. Summary:")
        for i, (time, memory) in enumerate(zip(times, peak_memories)):
            print(f"Run {i+1}: Time: {time:.2f}s, Peak Memory: {memory:.2f}GB")
        print(f"\nOverall Average Time: {avg_time:.2f}s")
        print(f"Overall Average Peak Memory: {avg_memory:.2f}GB")

def run_pab():
    config = LatteConfig("maxin-cn/Latte-1", enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")

if __name__ == "__main__":
    run_base()
    # run_pab()