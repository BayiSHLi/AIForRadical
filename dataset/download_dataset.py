import os
import subprocess
from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run(cmd):
    print(f"[Running] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# =========================
# 1. HuggingFace datasets
# =========================

def download_hf(name, save_path, subset=None):
    # Skip if already downloaded (dataset_info.json is written by save_to_disk)
    if os.path.exists(os.path.join(save_path, "dataset_info.json")) or \
       os.path.exists(os.path.join(save_path, "dataset_dict.json")):
        print(f"\n[Skipping HF dataset] {name}{f'/{subset}' if subset else ''} (already downloaded)")
        return

    ensure_dir(save_path)
    print(f"\n[Downloading HF dataset] {name}{f'/{subset}' if subset else ''}")

    if subset:
        ds = load_dataset(name, subset)
    else:
        ds = load_dataset(name)

    ds.save_to_disk(save_path)


def download_hf_datasets():
    # Toxicity / Hate — tweet_eval requires an explicit config name
    TWEET_EVAL_SUBSETS = ['emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment', 'stance_abortion', 'stance_atheism', 'stance_climate', 'stance_feminist', 'stance_hillary']
    for subset in TWEET_EVAL_SUBSETS:
        download_hf("tweet_eval", str(BASE_DIR / "tweet_eval" / subset), subset=subset)
    download_hf("hate_speech_offensive", str(BASE_DIR / "hatebase"))

    # OLID (offensive language)
    download_hf("christophsonntag/OLID", str(BASE_DIR / "olid"))

    # Jigsaw (toxicity)
    download_hf("civil_comments", str(BASE_DIR / "jigsaw"))

    # C4 English (⚠️ huge → only take first 10k examples via streaming)
    c4_path = BASE_DIR / "mc4_en_small"
    if os.path.exists(os.path.join(c4_path, "dataset_info.json")) or \
       os.path.exists(os.path.join(c4_path, "dataset_dict.json")):
        print(f"\n[Skipping HF dataset] allenai/c4/en (already downloaded)")
    else:
        ensure_dir(c4_path)
        print("\n[Downloading HF dataset] allenai/c4/en (streaming, first 10k rows)")
        from datasets import Dataset
        stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
        ds = Dataset.from_list(list(stream.take(10000)))
        ds.save_to_disk(str(c4_path))


# =========================
# 2. GitHub datasets
# =========================

def git_clone(repo_url, dest):
    if os.path.isdir(dest):
        print(f"[Skipping git clone] {dest} (already exists)")
        return
    run(f'git clone {repo_url} "{dest}"')


def download_github():
    # FakeNewsNet
    git_clone("https://github.com/KaiDMML/FakeNewsNet.git", str(BASE_DIR / "FakeNewsNet"))

    # CoAID
    git_clone("https://github.com/cuilimeng/CoAID.git", str(BASE_DIR / "CoAID"))

    # MIWS_Dataset
    git_clone("https://github.com/m4gaikwad/MIWS_Dataset_Standard.git", str(BASE_DIR / "MIWS_Dataset_Standard"))

# =========================
# 3. Personality datasets
# =========================

def download_personality_datasets():
    # Pandora Big-5 personality labels
    download_hf("jingjietan/pandora-big5", str(BASE_DIR / "pandora-big5"))


# =========================
# Main
# =========================

def main():
    ensure_dir(BASE_DIR)

    print("=== Step 1: HuggingFace datasets ===")
    download_hf_datasets()

    print("\n=== Step 2: GitHub datasets ===")
    download_github()

    print("\n=== Step 3: Personality datasets ===")
    download_personality_datasets()

    print("\n✅ All datasets downloaded to:", BASE_DIR)


if __name__ == "__main__":
    main()