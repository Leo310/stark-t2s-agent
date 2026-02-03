"""Download STaRK-Prime dataset from HuggingFace."""

import io
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from stark_prime_t2s.config import (
    PRIME_PROCESSED_DIR,
    PRIME_QA_HUMAN_URL,
    PRIME_QA_SYNTH_URL,
    PRIME_SKB_URL,
    QA_HUMAN_PATH,
    QA_SYNTH_PATH,
)


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest: Destination path
        desc: Description for progress bar
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  → {dest.name} already exists, skipping download")
        return
    
    print(f"  → Downloading {desc}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_and_extract_zip(url: str, extract_to: Path, desc: str = "Downloading") -> None:
    """Download a zip file and extract it.
    
    Args:
        url: URL to download from
        extract_to: Directory to extract to
        desc: Description for progress bar
    """
    # Check if already extracted
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"  → {extract_to.name}/ already exists, skipping download")
        return
    
    print(f"  → Downloading {desc}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    # Download to memory
    buffer = io.BytesIO()
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Download") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            buffer.write(chunk)
            pbar.update(len(chunk))
    
    # Extract
    print(f"  → Extracting to {extract_to}...")
    buffer.seek(0)
    with zipfile.ZipFile(buffer, "r") as zf:
        # Get the list of files for progress
        members = zf.namelist()
        extract_to.mkdir(parents=True, exist_ok=True)
        
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, extract_to.parent)


def download_prime_skb(force: bool = False) -> Path:
    """Download and extract STaRK-Prime SKB (processed bundle).
    
    Args:
        force: If True, re-download even if already exists
        
    Returns:
        Path to the processed directory
    """
    print("Downloading STaRK-Prime SKB...")
    
    if force and PRIME_PROCESSED_DIR.exists():
        import shutil
        shutil.rmtree(PRIME_PROCESSED_DIR)
    
    download_and_extract_zip(
        PRIME_SKB_URL,
        PRIME_PROCESSED_DIR,
        desc="STaRK-Prime SKB (processed.zip)"
    )
    
    # The zip extracts to a "processed" subdirectory
    # Make sure we have the expected structure
    expected_files = [
        "node_type_dict.pkl",
        "edge_type_dict.pkl",
        "node_info.pkl",
        "node_types.pt",
        "edge_index.pt",
        "edge_types.pt",
    ]
    
    for fname in expected_files:
        fpath = PRIME_PROCESSED_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected file not found after extraction: {fpath}\n"
                f"Contents of {PRIME_PROCESSED_DIR}: {list(PRIME_PROCESSED_DIR.iterdir()) if PRIME_PROCESSED_DIR.exists() else 'directory does not exist'}"
            )
    
    print(f"  ✓ STaRK-Prime SKB downloaded to {PRIME_PROCESSED_DIR}")
    return PRIME_PROCESSED_DIR


def download_prime_qa(force: bool = False) -> tuple[Path, Path]:
    """Download STaRK-Prime QA datasets (synthesized and human-generated).
    
    Args:
        force: If True, re-download even if already exists
        
    Returns:
        Tuple of (synth_path, human_path)
    """
    print("Downloading STaRK-Prime QA datasets...")
    
    if force:
        QA_SYNTH_PATH.unlink(missing_ok=True)
        QA_HUMAN_PATH.unlink(missing_ok=True)
    
    download_file(
        PRIME_QA_SYNTH_URL,
        QA_SYNTH_PATH,
        desc="Synthesized QA dataset"
    )
    
    download_file(
        PRIME_QA_HUMAN_URL,
        QA_HUMAN_PATH,
        desc="Human-generated QA dataset"
    )
    
    print(f"  ✓ QA datasets downloaded")
    return QA_SYNTH_PATH, QA_HUMAN_PATH


def download_all(force: bool = False) -> None:
    """Download all STaRK-Prime data.
    
    Args:
        force: If True, re-download even if already exists
    """
    download_prime_skb(force=force)
    download_prime_qa(force=force)
    print("\n✓ All STaRK-Prime data downloaded successfully!")


if __name__ == "__main__":
    download_all()

