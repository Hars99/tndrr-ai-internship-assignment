import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:1234"
CONTEXT_LENGTH = 2048
OPTIONS = {"temperature": 0, "max_output_tokens": 192, "store": False, "stream": False}

PROMPTS = [
    (
        "A bus leaves at 09:45 and travels for 2 hours 35 minutes. What time does it arrive? Give just the time.",
        "Correct time is 12:20. The response should contain only the time.",
    ),
    (
        "Write a Python function is_even(n) that tests integer parity. Return only the code.",
        "Valid Python, correct for even and odd integers, correct function name, and code-only output.",
    ),
    (
        "Return a JSON object with keys city and temperatures. The city is Guntur and the temperatures are 29, 31, and 30. Return only raw JSON.",
        "Valid JSON with the requested keys, Guntur as a string, and the three numbers as an array.",
    ),
    (
        "In exactly one sentence, explain why a laptop fan gets louder during a long video export.",
        "Exactly one sentence linking sustained processing, heat, and the cooling fan.",
    ),
    (
        "Classify this message as billing, technical, or account. Reply with one label. 'I was billed twice for this month's subscription.'",
        "Exactly the label billing.",
    ),
    (
        "Summarize in one sentence: The library will close for repairs on Monday and Tuesday. It will reopen Wednesday at 10 am. Books can be returned using the outside slot throughout the closure.",
        "Preserves closure dates, reopening time, and the outside return option.",
    ),
    (
        "Suggest exactly three short names for a mobile app that helps people track household water usage.",
        "Three distinct, short names relevant to water usage.",
    ),
    (
        "Rewrite in a polite but direct way, under 25 words: 'You sent the wrong spreadsheet again. Fix it before lunch.'",
        "Keeps the incorrect spreadsheet and deadline, stays polite, and uses fewer than 25 words.",
    ),
    (
        "The set A is {1, 2, 3, 5} and the set B is {2, 4, 5}. Which elements belong to exactly one of the two sets? Use ascending order.",
        "The elements are 1, 3, and 4, in that order.",
    ),
    (
        "A note says the project was approved on Tuesday and will begin next month. Who approved it? If the note does not say, reply 'Not specified.'",
        "Says Not specified without inventing an approver.",
    ),
]


def request(path, payload=None):
    headers = {"Content-Type": "application/json"}
    if token := os.getenv("LM_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(URL + path, data=data, headers=headers)
    try:
        with urlopen(req, timeout=180) as response:
            return json.load(response)
    except HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"LM Studio {path}: HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError("Start LM Studio's local server on port 1234 and try again.") from error


def lms_command():
    cli = shutil.which("lms")
    if cli:
        return cli
    bundled = Path.home() / ".lmstudio" / "bin" / "lms.exe"
    if bundled.is_file():
        return str(bundled)
    raise RuntimeError("Open LM Studio, then enable its lms command as shown in the README.")


def gpu_sample(index):
    output = subprocess.check_output(
        [
            "nvidia-smi", f"--id={index}",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True, timeout=8,
    )
    name, driver, total, used = next(csv.reader(output.splitlines(), skipinitialspace=True))
    return {"gpu": name, "driver": driver, "total_mib": int(total), "used_mib": int(used)}


def memory_samples(index):
    samples = []
    try:
        for _ in range(3):
            samples.append(gpu_sample(index))
            sleep(0.15)
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
        print("VRAM sampling unavailable. Record dedicated GPU memory from Task Manager for each variant.")
    return samples


def models():
    return request("/api/v1/models")["models"]


def verify_models(keys):
    available = {entry["key"]: entry for entry in models()}
    for key, allowed in zip(keys, ({"F16", "FP16", "FLOAT16"}, {"Q4_K_M"})):
        if key not in available:
            raise RuntimeError(f"Model key {key!r} not found. Check lms ls --detailed and pass its exact key.")
        name = (available[key].get("quantization") or {}).get("name")
        normalized = (name or "").upper().replace("-", "_").replace(" ", "")
        if normalized not in allowed:
            raise RuntimeError(f"Model {key!r} reports quantization {name!r}, expected {', '.join(sorted(allowed))}.")
    return available


def generate(instance, prompt):
    response = request("/api/v1/chat", {"model": instance, "input": prompt, **OPTIONS})
    if response["model_instance_id"] != instance:
        raise RuntimeError(f"LM Studio used {response['model_instance_id']!r}, expected {instance!r}.")
    answer = "\n".join(item["content"] for item in response["output"] if item["type"] == "message")
    return answer, response["stats"]


def run_model(cli, key, instance, index):
    idle = memory_samples(index)
    subprocess.run(
        [cli, "load", key, "--gpu", "max", "--context-length", str(CONTEXT_LENGTH), "--identifier", instance],
        check=True, timeout=120,
    )
    try:
        loaded = next(
            (entry for model in models() if model["key"] == key
             for entry in model["loaded_instances"] if entry["id"] == instance),
            None,
        )
        if loaded is None or loaded["config"]["context_length"] != CONTEXT_LENGTH:
            raise RuntimeError(f"{instance} must load with a {CONTEXT_LENGTH}-token context.")
        generate(instance, "Write a short sentence about a bicycle.")
        warm = memory_samples(index)
        outputs = []
        for i, (prompt, review) in enumerate(PROMPTS, 1):
            start = perf_counter()
            answer, stats = generate(instance, prompt)
            elapsed = perf_counter() - start
            rate = stats.get("tokens_per_second")
            if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                rate = None
            outputs.append({
                "prompt_id": i, "prompt": prompt, "quality_check": review,
                "output": answer, "request_seconds": elapsed,
                "input_tokens": stats.get("input_tokens"),
                "output_tokens": stats.get("total_output_tokens"),
                "tokens_per_second": rate,
                "time_to_first_token_seconds": stats.get("time_to_first_token_seconds"),
            })
            speed = f"{rate:.1f} tokens/s" if rate is not None else "rate unavailable"
            print(f"  {i:02d}/{len(PROMPTS)}  {speed}")
        idle_mib = median(sample["used_mib"] for sample in idle) if idle else None
        loaded_mib = median(sample["used_mib"] for sample in warm) if warm else None
        vram_delta = loaded_mib - idle_mib if idle_mib is not None and loaded_mib is not None else None
        return {
            "model_key": key, "model_instance": instance, "load_config": loaded["config"],
            "gpu_offload_requested": "max", "nvidia_smi_idle_samples": idle, "nvidia_smi_warm_samples": warm,
            "idle_gpu_mib": idle_mib, "warm_gpu_mib": loaded_mib, "warm_minus_idle_gpu_mib": vram_delta,
            "outputs": outputs,
        }
    finally:
        try:
            request("/api/v1/models/unload", {"instance_id": instance})
        except RuntimeError as error:
            print(f"Unload {instance} in LM Studio before the next model if it is still loaded: {error}")


def write_outputs(runs):
    path = Path("benchmark_outputs.md")
    with path.open("w", encoding="utf-8") as output:
        output.write("# Output comparison\n\n")
        for i, (prompt, review) in enumerate(PROMPTS, 1):
            output.write(f"## Prompt {i}\n\n{prompt}\n\nCheck: {review}\n\n")
            for run in runs:
                output.write(f"### {run['model_key']}\n\n{run['outputs'][i - 1]['output']}\n\n")
            output.write("Your observation:\n\n")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two Qwen2.5 GGUF variants using LM Studio.")
    parser.add_argument("--fp16-key", default="tndrr/qwen25-05b-fp16")
    parser.add_argument("--q4-key", default="tndrr/qwen25-05b-q4")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    cli = lms_command()
    keys = (args.fp16_key, args.q4_key)
    available = verify_models(keys)
    already_loaded = [instance["id"] for model in available.values() for instance in model["loaded_instances"]]
    if already_loaded:
        print("Already loaded (these can affect memory or speed):", ", ".join(already_loaded))
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "options": OPTIONS, "backend": "LM Studio native API with llama.cpp GGUF models",
        "requested_context_length": CONTEXT_LENGTH, "gpu_index_sampled": args.gpu_index,
        "loaded_before_run": already_loaded, "runs": [],
    }
    out = Path("benchmark_results.json")
    for key, instance in zip(keys, ("tndrr-fp16", "tndrr-q4")):
        print(f"\n{key}")
        run = run_model(cli, key, instance, args.gpu_index)
        run["quantization"] = available[key]["quantization"]
        run["gguf_size_bytes"] = available[key]["size_bytes"]
        result["runs"].append(run)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if run["warm_minus_idle_gpu_mib"] is not None:
            print(f"Warm VRAM above idle: {run['warm_minus_idle_gpu_mib']:.0f} MiB")
    paired = [
        (a, b) for a, b in zip(*(run["outputs"] for run in result["runs"]))
        if a["tokens_per_second"] is not None and b["tokens_per_second"] is not None
    ]
    result["throughput_prompt_ids"] = [a["prompt_id"] for a, _ in paired]
    for index, run in enumerate(result["runs"]):
        if paired:
            mean_rate = fmean(pair[index]["tokens_per_second"] for pair in paired)
            run["mean_decode_tokens_per_second"] = mean_rate
            print(f"{run['quantization']['name']}: {mean_rate:.1f} tokens/s over {len(paired)} matched prompts")
        else:
            print("No prompts had a usable decode rate for both variants.")
            break
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")
    write_outputs(result["runs"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, KeyError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Error: {error}") from None
