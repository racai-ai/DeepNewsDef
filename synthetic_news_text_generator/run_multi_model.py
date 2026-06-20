#!/usr/bin/env python3
"""
run_multi_model.py
==================
Multi-model synthetic news generation script for DeepNewsDef.

Runs the generation pipeline across every LLM defined in config.yaml,
distributing the requested article count across all models (round-robin)
or focusing on a specific subset of models.

Usage examples
--------------
# Generate 50 articles distributed across all models in config.yaml
python run_multi_model.py --count 50

# Generate 100 articles using only two specific models
python run_multi_model.py --count 100 --models "llama3.2:3b" "mistral:7b"

# Generate 20 articles per model (instead of distributing a total)
python run_multi_model.py --per-model 20

# Run with 4 parallel workers (one thread per model)
python run_multi_model.py --count 200 --workers 4

# Sequential mode: delete each model from disk after its batch (saves disk space)
python run_multi_model.py --per-model 30 --delete-after

# Dry-run: show the generation plan without calling Ollama
python run_multi_model.py --count 20 --dry-run

# Disable Wikipedia seeding (avoid rate limits)
python run_multi_model.py --per-model 10 --no-wiki
"""

import os
import sys
import argparse
import random
import json
import time
import threading
import traceback
import subprocess
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root (parent of this package folder) is on sys.path
# so that `synthetic_news_text_generator.*` imports resolve correctly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # d:\DeepNewsDef
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_news_text_generator.config_parser import load_config, LLMConfig
from synthetic_news_text_generator.wiki_scraper import fetch_random_wikipedia_extract
from synthetic_news_text_generator.prompt_engine import PromptEngine
from synthetic_news_text_generator.llm_interface import LLMInterface
from synthetic_news_text_generator.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Thread-safe progress tracker
# ---------------------------------------------------------------------------
class ProgressTracker:
    def __init__(self, total: int):
        self._lock = threading.Lock()
        self.total = total
        self.done = 0
        self.failed = 0
        self.model_stats: dict[str, dict] = defaultdict(lambda: {"success": 0, "fail": 0, "time_s": []})
        self.start_time = time.time()

    def record_success(self, model: str, elapsed: float):
        with self._lock:
            self.done += 1
            self.model_stats[model]["success"] += 1
            self.model_stats[model]["time_s"].append(elapsed)
            self._print_progress()

    def record_failure(self, model: str, reason: str):
        with self._lock:
            self.failed += 1
            self.model_stats[model]["fail"] += 1
            print(f"  [FAIL] {model}: {reason}")
            self._print_progress()

    def _print_progress(self):
        pct = (self.done + self.failed) / self.total * 100
        elapsed = time.time() - self.start_time
        rate = (self.done + self.failed) / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done - self.failed) / rate if rate > 0 else float("inf")
        eta_str = f"{eta:.0f}s" if eta != float("inf") else "?"
        print(
            f"  Progress: {self.done + self.failed}/{self.total} ({pct:.1f}%) | "
            f"✓ {self.done}  ✗ {self.failed} | "
            f"rate={rate:.2f}/s ETA={eta_str}",
            flush=True,
        )

    def print_summary(self):
        total_elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print(f"  GENERATION COMPLETE")
        print("=" * 70)
        print(f"  Total articles:  {self.total}")
        print(f"  ✓ Succeeded:     {self.done}")
        print(f"  ✗ Failed:        {self.failed}")
        print(f"  Elapsed:         {total_elapsed:.1f}s")
        print()
        print("  Per-model breakdown:")
        for model, stats in sorted(self.model_stats.items()):
            times = stats["time_s"]
            avg_t = sum(times) / len(times) if times else 0
            print(
                f"    {model:<30}  ✓={stats['success']:>4}  ✗={stats['fail']:>3}"
                f"  avg_time={avg_t:.1f}s"
            )
        print("=" * 70)


# ---------------------------------------------------------------------------
# Core generation job
# ---------------------------------------------------------------------------
def build_generation_job(config, llm_config: LLMConfig, api_key: str, use_wiki_override: bool | None = None):
    """Pick random parameters and build a single generation job dict."""
    model_display_name = llm_config.display_name or llm_config.name
    country = random.choice(config.target_countries)
    persona = random.choice(config.personas)
    length = random.choice(config.lengths)
    platform = random.choice(config.platforms)
    viewpoint = random.choice(config.viewpoints)
    grammar = random.choice(config.grammar_levels)
    manipulation = random.choice(config.manipulation_levels)
    propaganda = random.choice(config.propaganda_techniques)
    audience = random.choice(config.target_audiences)
    urgency = random.choice(config.urgency_levels)
    formatting = random.choice(config.formatting_styles)

    topic_info = random.choice(config.topics)
    domain = topic_info.domain
    subdomain = random.choice(topic_info.subdomains)
    topic_str = f"{domain} - {subdomain}"

    temperature = random.choice(llm_config.temperature_range)
    top_p = random.choice(llm_config.top_p_range)
    frequency_penalty = random.choice(llm_config.frequency_penalty_range)
    presence_penalty = random.choice(llm_config.presence_penalty_range)
    max_tokens = llm_config.max_tokens

    # Decide generation type
    use_seed = getattr(config, "use_wikipedia_seed", True)
    if use_wiki_override is not None:
        use_seed = use_wiki_override
    if use_seed == "random":
        use_seed = random.choice([True, False])

    gen_types = ["complete", "based_on_topic"]
    if use_seed:
        gen_types.append("based_on_real_article")
    generation_type = random.choice(gen_types)

    seed = ""
    seed_url = ""
    if generation_type == "based_on_real_article":
        try:
            seed, seed_url = fetch_random_wikipedia_extract()
            if not seed:
                raise ValueError("Empty Wikipedia extract")
        except Exception as exc:
            print(f"  [WARN] Wiki fetch failed ({exc}). Falling back to 'complete'.")
            generation_type = "complete"

    # Build prompt
    if generation_type == "complete":
        prompt = PromptEngine.build_complete_prompt(
            persona, country, length, platform, topic_str, viewpoint,
            grammar, manipulation, propaganda, audience, urgency, formatting
        )
    elif generation_type == "based_on_real_article":
        prompt = PromptEngine.build_based_on_real_article_prompt(
            persona, country, length, platform, viewpoint, seed,
            grammar, manipulation, propaganda, audience, urgency, formatting
        )
    else:  # based_on_topic
        prompt = PromptEngine.build_based_on_topic_prompt(
            persona, country, length, topic_str, viewpoint,
            grammar, manipulation, propaganda, audience, urgency, formatting
        )

    return {
        "model_name": llm_config.name,
        "model_display_name": model_display_name,
        "endpoint": llm_config.endpoint,
        "quantization": llm_config.quantization,
        "api_key": api_key,
        "system_prompt": PromptEngine.get_system_prompt(),
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "country": country,
        "generation_type": generation_type,
        "topic": topic_str,
        "domain": domain,
        "subdomain": subdomain,
        "persona": persona,
        "seed_url": seed_url,
    }


# ---------------------------------------------------------------------------
# Model deletion
# ---------------------------------------------------------------------------
def delete_model(model_name: str) -> bool:
    """
    Delete an Ollama model from disk using `ollama rm <model>`.
    Returns True on success, False if the command fails.
    """
    print(f"  [DELETE] Removing {model_name} from disk...", flush=True)
    try:
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            print(f"  [DELETE] ✓ {model_name} removed.", flush=True)
            return True
        else:
            print(
                f"  [DELETE] ✗ Failed to remove {model_name}: {result.stderr.strip()}",
                flush=True,
            )
            return False
    except FileNotFoundError:
        print("  [DELETE] ✗ 'ollama' command not found. Is Ollama installed and on PATH?", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"  [DELETE] ✗ Timeout while removing {model_name}.", flush=True)
        return False


def pull_model(model_name: str) -> bool:
    """
    Pull an Ollama model to local disk using `ollama pull <model>`.
    Returns True on success, False if the command fails.
    """
    print(f"  [PULL] Downloading {model_name}...", flush=True)
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=60 * 60,
        )
        if result.returncode == 0:
            print(f"  [PULL] ✓ {model_name} ready.", flush=True)
            return True
        else:
            print(
                f"  [PULL] ✗ Failed to pull {model_name}: {result.stderr.strip()}",
                flush=True,
            )
            return False
    except FileNotFoundError:
        print("  [PULL] ✗ 'ollama' command not found. Is Ollama installed and on PATH?", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"  [PULL] ✗ Timeout while pulling {model_name}.", flush=True)
        return False


def execute_job(job: dict, output_dir: str, tracker: ProgressTracker, retries: int = 2):
    """Execute a single generation job, saving outputs to disk."""
    model_name = job["model_name"]
    model_display_name = job["model_display_name"]
    t0 = time.time()

    llm_interface = LLMInterface(api_key=job["api_key"], endpoint=job["endpoint"])
    pipeline = Pipeline(llm_interface, model_name=model_name)

    last_error = None
    for attempt in range(1, retries + 2):  # attempts = retries + 1
        try:
            article = llm_interface.generate_article(
                model=model_name,
                system_prompt=job["system_prompt"],
                user_prompt=job["prompt"],
                temperature=job["temperature"],
                top_p=job["top_p"],
                frequency_penalty=job["frequency_penalty"],
                presence_penalty=job["presence_penalty"],
            )

            context = {
                "country": job["country"],
                "generation_type": job["generation_type"],
                "prompt": job["prompt"],
                "model": model_display_name,
                "quantization": job["quantization"],
                "temperature": job["temperature"],
                "top_p": job["top_p"],
                "frequency_penalty": job["frequency_penalty"],
                "presence_penalty": job["presence_penalty"],
                "max_tokens": job["max_tokens"],
                "topic": job["topic"],
                "domain": job["domain"],
                "subdomain": job["subdomain"],
                "persona": job["persona"],
            }

            metadata = pipeline.process(article, context)
            metadata.generation.paragraphs = []

            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"{metadata.id}.json")
            txt_path = os.path.join(output_dir, f"{metadata.id}.txt")

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(metadata.model_dump_json(by_alias=True, indent=4))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(article.title + "\n\n" + "\n\n".join(article.paragraphs))

            elapsed = time.time() - t0
            tracker.record_success(model_name, elapsed)
            return True

        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                wait = 2 ** attempt  # exponential back-off: 2s, 4s, ...
                print(f"  [RETRY {attempt}/{retries}] {model_name}: {exc}. Waiting {wait}s...")
                time.sleep(wait)

    tracker.record_failure(model_name, str(last_error))
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-model synthetic news generation for DeepNewsDef",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml, resolved relative to the module directory)",
    )
    # Mutually exclusive count modes
    count_group = parser.add_mutually_exclusive_group(required=True)
    count_group.add_argument(
        "--count",
        type=int,
        metavar="N",
        help="Total number of articles to generate, distributed across all models.",
    )
    count_group.add_argument(
        "--per-model",
        type=int,
        metavar="N",
        dest="per_model",
        help="Number of articles to generate per model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Restrict generation to these model names (must match names in config.yaml).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads (default: 1 = sequential). "
             "Use with care; each worker keeps a model busy.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retry attempts on failure (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Override the output directory from config.yaml.",
    )
    parser.add_argument(
        "--delete-after",
        action="store_true",
        dest="delete_after",
        help="Sequential mode only: delete each model from disk with 'ollama rm' "
             "after its batch of jobs is complete. Frees disk space before the next "
             "model is pulled. Has no effect when --workers > 1.",
    )
    parser.add_argument(
        "--pull-before",
        action="store_true",
        dest="pull_before",
        help="Sequential mode only: pull each model with 'ollama pull' before its batch. "
             "Forces grouped-by-model execution. Has no effect when --workers > 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Build the job list and print the plan without calling Ollama.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-wiki",
        action="store_true",
        dest="no_wiki",
        help="Disable Wikipedia seeding (avoids rate limits). Overrides config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[INFO] Random seed set to {args.seed}")

    # --- Load config ---
    config = load_config(args.config)

    # --- Filter models ---
    available_models = config.llms
    if args.models:
        requested = set(args.models)
        available_models = [m for m in config.llms if m.name in requested]
        missing = requested - {m.name for m in available_models}
        if missing:
            print(f"[WARN] Models not found in config.yaml and will be skipped: {missing}")
        if not available_models:
            print("[ERROR] No matching models found. Exiting.")
            sys.exit(1)

    print(f"[INFO] Using {len(available_models)} model(s): {[m.name for m in available_models]}")

    # --- Determine output directory ---
    output_dir = args.output_dir or config.output_directory
    output_dir = os.path.join(PROJECT_ROOT, output_dir) if not os.path.isabs(output_dir) else output_dir
    print(f"[INFO] Output directory: {output_dir}")

    # --- Build job list ---
    api_key = os.getenv("OPENAI_API_KEY", "ollama_dummy_key")
    use_wiki_override = False if args.no_wiki else None

    if args.per_model:
        # N articles per each model
        jobs = []
        for llm_config in available_models:
            for _ in range(args.per_model):
                jobs.append(build_generation_job(config, llm_config, api_key, use_wiki_override))
    else:
        # Distribute total count round-robin across models
        total = args.count
        jobs = []
        for idx in range(total):
            llm_config = available_models[idx % len(available_models)]
            jobs.append(build_generation_job(config, llm_config, api_key, use_wiki_override))

    group_by_model = args.delete_after or args.pull_before
    if not group_by_model:
        random.shuffle(jobs)  # avoid locality effects in parallel mode
    total_jobs = len(jobs)

    print(f"[INFO] Total generation jobs: {total_jobs}")

    # --- Dry-run: just print the plan ---
    if args.dry_run:
        model_counts: dict[str, int] = defaultdict(int)
        gen_type_counts: dict[str, int] = defaultdict(int)
        for j in jobs:
            model_counts[j["model_name"]] += 1
            gen_type_counts[j["generation_type"]] += 1

        print("\n[DRY RUN] Generation plan:")
        for model, cnt in sorted(model_counts.items()):
            print(f"  {model:<30}  {cnt} articles")
        print("\n  Generation type distribution:")
        for gt, cnt in sorted(gen_type_counts.items()):
            print(f"  {gt:<30}  {cnt}")
        print("\n[DRY RUN] No articles were generated. Remove --dry-run to execute.")
        return

    # --- Execute ---
    tracker = ProgressTracker(total=total_jobs)

    print(f"\n[START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"        workers={args.workers}  retries={args.retries}  "
        f"pull_before={args.pull_before}  delete_after={args.delete_after}\n"
    )

    if args.workers <= 1:
        if group_by_model:
            # Group jobs by model order so all of model-A runs before model-B, etc.
            jobs_by_model: dict[str, list] = defaultdict(list)
            for job in jobs:
                jobs_by_model[job["model_name"]].append(job)

            for llm_config in available_models:
                model_jobs = jobs_by_model.get(llm_config.name, [])
                if not model_jobs:
                    continue
                print(f"\n[BATCH] {llm_config.name}  ({len(model_jobs)} jobs)", flush=True)
                if args.pull_before:
                    if not pull_model(llm_config.name):
                        for _ in model_jobs:
                            tracker.record_failure(llm_config.name, "pull failed")
                        continue
                for job in model_jobs:
                    execute_job(job, output_dir, tracker, retries=args.retries)
                if args.delete_after:
                    delete_model(llm_config.name)
        else:
            # Standard sequential — jobs were already shuffled above
            for job in jobs:
                execute_job(job, output_dir, tracker, retries=args.retries)
    else:
        if args.pull_before:
            print("[WARN] --pull-before is ignored in parallel mode (--workers > 1).")
        if args.delete_after:
            print("[WARN] --delete-after is ignored in parallel mode (--workers > 1).")
        # Parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(execute_job, job, output_dir, tracker, args.retries): job
                for job in jobs
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    job = futures[future]
                    print(f"  [UNCAUGHT] {job['model_name']}: {exc}")
                    traceback.print_exc()

    tracker.print_summary()


if __name__ == "__main__":
    main()
