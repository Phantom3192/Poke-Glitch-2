"""
train_model.py - Stream Processing AI Pokémon Trainer
Trains in batches, clears memory after each batch, saves to DB incrementally
"""

import os
import sys
import json
import logging
import sqlite3
import random
import time
import zipfile
import shutil
import subprocess
import gc
import resource
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from io import BytesIO

# ============ NUCLEAR LOG SUPPRESSION ============
logging.root.handlers = []
logging.basicConfig = lambda *args, **kwargs: None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except:
    pass

for name in logging.root.manager.loggerDict.keys():
    logging.getLogger(name).disabled = True
    logging.getLogger(name).setLevel(logging.CRITICAL)

sys.stderr = open(os.devnull, 'w') if not os.getenv("DEBUG") else sys.stderr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm

# ============ SILENT LOGGER ============
class SilentLogger:
    def info(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO {msg}")
    def warning(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} WARNING {msg}")
    def error(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR {msg}")
    def debug(self, msg, *args, **kwargs):
        pass

log = SilentLogger()

def log_memory(tag: str = ""):
    """
    Logs current process RSS (resident memory) using stdlib `resource`
    - no extra dependency. Lets you actually watch memory climb in the
    Railway logs batch-by-batch instead of only finding out when the
    container gets killed. ru_maxrss is KB on Linux.
    """
    try:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log.info(f"   🧠 Memory{f' ({tag})' if tag else ''}: {rss_mb:.0f} MB peak RSS")
    except Exception:
        pass


class PrefetchIterator:
    """
    Wraps a slow iterator (e.g. one that hits the network) in a background
    thread with a small bounded queue, so the NEXT item can be fetched
    while the caller is busy doing something else (e.g. training on the
    current chunk) - overlapping network wait time with CPU compute time
    instead of paying both costs back-to-back.

    Deliberately conservative about not crashing the main process:
    - queue is bounded (maxsize) so a fast producer can't outrun the
      consumer and blow up memory
    - the background thread is a daemon, so it can never block shutdown
    - any exception in the producer is caught, stored, and re-raised in
      the MAIN thread on next() - it never crashes silently or hangs
    - if thread startup itself fails for any reason, falls back to
      plain synchronous iteration (no prefetching, but still works)
    """
    _SENTINEL = object()

    def __init__(self, source_iterable, maxsize: int = 4):
        self._source = source_iterable
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        try:
            self._thread = threading.Thread(target=self._produce, daemon=True)
            self._thread.start()
        except Exception as e:
            log.warning(f"   ⚠️ Prefetch thread failed to start ({e}), "
                        f"falling back to non-prefetched loading")
            self._thread = None

    def _produce(self):
        try:
            for item in self._source:
                self._queue.put(item)
        except BaseException as e:
            self._error = e
        finally:
            self._queue.put(self._SENTINEL)

    def __iter__(self):
        if self._thread is None:
            # Fallback: no thread running, just iterate the source directly.
            yield from self._source
            return
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            yield item

# ============ CONFIGURATION ============

TURSO_URL = os.getenv("TURSO_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Increased for streaming
STREAM_BATCH_SIZE = int(os.getenv("STREAM_BATCH_SIZE", "20"))  # Images per stream batch - kept small so only one small chunk is ever in memory at a time
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
EARLY_STOP_ACC = float(os.getenv("EARLY_STOP_ACC", "95.0"))  # stop once train acc hits this %
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "3"))  # stop if val acc doesn't improve for N epochs
DATASET_NAME = os.getenv("DATASET_NAME", "SpreadSheets/Poketwo-Spawn-Images")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
# Full training state (model + optimizer + scheduler + epoch/best-acc
# bookkeeping), written at the end of every epoch. This is what lets a
# restart resume training instead of starting over with fresh random
# weights - MODEL_OUTPUT above only ever held feature_extractor weights
# and was write-only (nothing ever loaded it back in).
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "models/checkpoint.pt")
DB_PATH = os.getenv("DB_PATH", "pokemon.db")
AUTO_EXTRACT_ARCHIVES = os.getenv("AUTO_EXTRACT_ARCHIVES", "true").lower() == "true"
MAX_SPECIES = int(os.getenv("MAX_SPECIES", "100"))  # Max species to train
MAX_IMAGES_PER_SPECIES = int(os.getenv("MAX_IMAGES_PER_SPECIES", "10"))
# Hard cap on how many images get cached in RAM for reuse across epochs.
# Each cached image is a 224x224x3 uint8 tensor (~150KB). Default 8000
# images caps the cache at ~1.2GB, which should be safe on most small
# Railway instances. Images beyond this cap still get used for training
# in epoch 1, they just aren't kept in memory - so if you hit the cap,
# epoch 2+ will train on a smaller (but still random/representative)
# subset instead of the full dataset. Raise this if your container has
# more RAM to spare; set to 0 to disable caching entirely (every epoch
# re-streams from Hugging Face - slower, but flat memory usage).
MAX_CACHE_IMAGES = int(os.getenv("MAX_CACHE_IMAGES", "0"))

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    log.info(f"🔑 HF_TOKEN configured")

DEVICE = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)

# ============ ARCHIVE EXTRACTION ============

def extract_archive_files():
    if not AUTO_EXTRACT_ARCHIVES:
        return
    
    archives = []
    for pattern in ["*.zip", "*.ZIP", "*.rar", "*.RAR", "*.7z", "*.7Z"]:
        archives.extend(Path(".").glob(pattern))
    
    if not archives:
        log.info("📦 No archive files found.")
        return
    
    log.info(f"📦 Found {len(archives)} archive file(s), extracting...")
    
    extra_dir = Path("Extra pokemons")
    extra_dir.mkdir(exist_ok=True)
    
    extracted_count = 0
    
    for archive_path in archives:
        try:
            ext = archive_path.suffix.lower()
            temp_dir = Path(f"temp_extract_{archive_path.stem}")
            temp_dir.mkdir(exist_ok=True)
            
            if ext in ['.zip']:
                log.info(f"   📂 Extracting ZIP: {archive_path.name}")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif ext in ['.rar']:
                log.info(f"   📂 Extracting RAR: {archive_path.name}")
                try:
                    import rarfile
                    with rarfile.RarFile(archive_path) as rf:
                        rf.extractall(temp_dir)
                except:
                    subprocess.run(['unrar', 'x', '-y', str(archive_path), str(temp_dir)], 
                                 capture_output=True, check=False)
            elif ext in ['.7z']:
                log.info(f"   📂 Extracting 7z: {archive_path.name}")
                try:
                    import py7zr
                    with py7zr.SevenZipFile(archive_path, 'r') as sz:
                        sz.extractall(temp_dir)
                except:
                    subprocess.run(['7z', 'x', '-y', str(archive_path), f'-o{temp_dir}'], 
                                 capture_output=True, check=False)
            
            extracted = process_extracted_files(temp_dir, extra_dir)
            extracted_count += extracted
            
            shutil.rmtree(temp_dir)
            archive_path.unlink()
            log.info(f"   ✅ Extracted: {archive_path.name} ({extracted} images)")
            
        except Exception as e:
            log.error(f"   ❌ Failed to extract {archive_path}: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    if extra_dir.exists():
        species = [f for f in extra_dir.iterdir() if f.is_dir()]
        if species:
            log.info(f"   📁 Extracted {len(species)} species, {extracted_count} images total")


def process_extracted_files(temp_dir: Path, extra_dir: Path) -> int:
    extracted_count = 0
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP'}
    
    for root, dirs, files in os.walk(temp_dir):
        root_path = Path(root)
        images = [f for f in files if Path(f).suffix in valid_extensions]
        
        if images:
            rel_path = root_path.relative_to(temp_dir)
            species_name = rel_path.parts[0].replace("_", " ").strip() if rel_path.parts else "unknown"
            
            dest_dir = extra_dir / species_name.replace(" ", "_")
            dest_dir.mkdir(exist_ok=True)
            
            for img_name in images:
                src = root_path / img_name
                dest = dest_dir / img_name
                if dest.exists():
                    counter = 1
                    stem = Path(img_name).stem
                    suffix = Path(img_name).suffix
                    while dest.exists():
                        dest = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.move(str(src), str(dest))
                extracted_count += 1
    
    return extracted_count

# ============ DATABASE LAYER ============

class Database:
    def __init__(self):
        self.use_turso = False
        self._conn = None
        
        if TURSO_URL:
            try:
                import libsql_experimental as libsql
                self._conn = libsql.connect(TURSO_URL, auth_token=TURSO_AUTH_TOKEN)
                self.use_turso = True
                log.info(f"✅ Connected to Turso database")
            except Exception:
                log.warning(f"Turso connection failed, using SQLite fallback")
        
        if not self.use_turso:
            self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            log.info(f"✅ Using SQLite database: {DB_PATH}")
        
        self._create_tables()
    
    def _create_tables(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pokemon_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                feature_vector TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(species, variant_name)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS species_info (
                species TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0,
                last_updated INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        self._conn.commit()
        log.info("✅ Database tables ready")
    
    def add_pokemon_features(self, species: str, features: List[np.ndarray], 
                             variant_names: List[str] = None):
        self.add_pokemon_features_batch({species: features})

    def add_pokemon_features_batch(self, species_to_features: Dict[str, List[np.ndarray]]):
        """
        Writes features for MULTIPLE species in ONE cursor + ONE commit,
        instead of one cursor/commit per species. This matters a lot once
        we're saving after every small training chunk (a chunk of ~25
        images can easily span 15-25 different species) - doing a separate
        commit per species per chunk multiplies DB round-trips a lot, and
        with the experimental Turso client in particular that repeated
        cursor/commit churn is a plausible source of memory growth over
        many batches. Batching it into one commit per chunk avoids that.

        variant_name is offset by each species' EXISTING row count so
        chunks accumulate (species_11, species_12, ...) instead of every
        chunk restarting at species_1 and silently INSERT-OR-REPLACE-ing
        over an earlier chunk's rows for the same species.
        """
        if not species_to_features:
            return
        cursor = self._conn.cursor()
        try:
            for species, features in species_to_features.items():
                if not features:
                    continue
                cursor.execute(
                    "SELECT COUNT(*) FROM pokemon_features WHERE species = ?", (species,)
                )
                existing_count = cursor.fetchone()[0]
                variant_names = [f"{species}_{existing_count + i + 1}" for i in range(len(features))]
                for i, feature in enumerate(features):
                    feature_json = json.dumps(feature.tolist())
                    cursor.execute("""
                        INSERT OR REPLACE INTO pokemon_features 
                        (species, variant_name, feature_vector, created_at)
                        VALUES (?, ?, ?, strftime('%s', 'now'))
                    """, (species, variant_names[i], feature_json))
                cursor.execute("""
                    INSERT INTO species_info (species, count, last_updated)
                    VALUES (?, ?, strftime('%s', 'now'))
                    ON CONFLICT(species) DO UPDATE SET
                        count = count + excluded.count,
                        last_updated = excluded.last_updated
                """, (species, len(features)))
            self._conn.commit()
        finally:
            try:
                cursor.close()
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pokemon_features")
        total_features = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM species_info")
        total_species = cursor.fetchone()[0]
        return {"total_features": total_features, "total_species": total_species, "use_turso": self.use_turso}
    
    def close(self):
        if self._conn:
            self._conn.close()


# ============ AI MODEL ============

class PokemonFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        # ShuffleNetV2 x0.5: fastest of the reasonable CPU-friendly options
        # (1.4M params vs MobileNetV3-Small's 2.5M) - swapped in specifically
        # for epoch speed now that the disk cache means epochs 2+ are
        # compute-bound rather than network-bound. Trade-off: its features
        # are somewhat weaker than MobileNetV3-Small's, which matters more
        # than usual here given how little data there is per species (25
        # images x 1119 classes) - worth watching train/val accuracy after
        # this swap to confirm it's not costing more than the speed is worth.
        self.backbone = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        backbone_dim = 1024  # ShuffleNetV2 x0.5's pre-fc feature dim
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.to(DEVICE)
        self.eval()
    
    @torch.no_grad()
    def extract(self, img: Image.Image) -> np.ndarray:
        if img is None:
            return np.zeros(256)
        
        try:
            img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
            img_tensor = self.normalize(img_tensor)
            # backbone already returns a flat (N, backbone_dim) vector —
            # its forward() does avgpool+flatten internally before the
            # (now-identity) classifier. Do NOT re-pool here; the old
            # code's adaptive_avg_pool2d on an already-2D tensor was
            # silently throwing and falling back to an all-zero vector
            # on every single image.
            features = self.backbone(img_tensor)
            projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)
            return projected.cpu().numpy().flatten()
        except Exception as e:
            log.warning(f"   ⚠️ Feature extraction failed: {e}")
            return np.zeros(256)
    
    def extract_batch(self, images: List[Image.Image], grad: bool = False) -> np.ndarray:
        """
        grad=False (default): used everywhere at inference time (bot lookups,
        DB feature snapshots) - whole thing runs under no_grad, nothing trains.

        grad=True: used during training. The backbone is still frozen/no_grad
        (it's pretrained ImageNet weights we don't want to disturb), but
        `projection` runs WITH grad so its weights actually receive gradient
        updates. Previously this whole method - backbone AND projection - ran
        under @torch.no_grad(), so `projection` never trained and stayed at
        its random init for the entire run while only the final classifier
        head learned on top of that noise. That's a big part of why accuracy
        was stuck low.
        """
        valid_images = []
        for img in images:
            if img is not None and isinstance(img, Image.Image):
                try:
                    valid_images.append(self.transform(img).unsqueeze(0))
                except Exception:
                    continue
        
        if not valid_images:
            return np.zeros((len(images), 256))
        
        try:
            batch_tensor = torch.cat(valid_images, dim=0).to(DEVICE)
            batch_tensor = self.normalize(batch_tensor)

            with torch.no_grad():
                features = self.backbone(batch_tensor)  # frozen, already flat (N, backbone_dim)

            if grad:
                projected = self.projection(features)
            else:
                with torch.no_grad():
                    projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)

            if grad:
                # keep it a tensor with grad history for the training loop
                result = projected
                if len(valid_images) < len(images):
                    pad = torch.zeros(len(images) - len(valid_images), result.shape[1], device=result.device)
                    result = torch.cat([result, pad], dim=0)
                return result

            result = projected.cpu().numpy()
            if len(valid_images) < len(images):
                padded = np.zeros((len(images), result.shape[1]))
                padded[:len(valid_images)] = result
                return padded
            return result
        except Exception as e:
            log.warning(f"   ⚠️ Batch feature extraction failed: {e}")
            if grad:
                return torch.zeros(len(images), 256, device=DEVICE, requires_grad=True)
            return np.zeros((len(images), 256))


class PokemonClassifier(nn.Module):
    def __init__(self, num_species: int):
        super().__init__()
        self.feature_extractor = PokemonFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_species)
        )
    
    def forward_batch(self, images: List[Image.Image], grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor.extract_batch(images, grad=grad)
        if grad:
            # already a tensor with grad history - don't detach it via torch.tensor()
            features_tensor = features.to(DEVICE)
        else:
            features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        logits = self.classifier(features_tensor)
        return features_tensor, logits


# ============ STREAMING DATASET ============

class StreamingPokemonDataset(IterableDataset):
    """
    Streams images from Hugging Face in batches.
    Processes STREAM_BATCH_SIZE images, trains, then clears memory.
    """
    
    def __init__(self, extra_dir: str = "Extra pokemons"):
        self.extra_dir = extra_dir
        # NOTE: intentionally stops short of ToTensor+Normalize here. Every
        # image produced by this transform gets cached in self._cache for
        # reuse across epochs (see __iter__), so we store it as a uint8
        # tensor (raw 0-255 pixels) instead of a normalized float32 tensor.
        # That's a 4x memory cut (1 byte/channel vs 4 bytes/channel) with
        # ~28k images this is the difference between ~16.8GB and ~4.2GB
        # of cache. Normalization is applied later, per-batch, right before
        # the forward pass (see _to_normalized_float below).
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.PILToTensor(),  # uint8, CHW, values 0-255
        ])
        
        # Cache of (tensor, label) collected on the first pass, reused for
        # every later epoch so we never re-hit Hugging Face after epoch 1.
        self._cache: List[Tuple[torch.Tensor, int]] = []
        self._cache_ready = False
        
        # Disk-backed chunk cache: epoch 1 writes (tensor,label) pairs to
        # small files on disk as it streams from HF, instead of only
        # keeping them in RAM. Epoch 2+ then reads those chunk files
        # straight off disk - no network calls, no RAM buildup (one small
        # chunk loaded at a time), and it even survives a container
        # restart. This is independent of MAX_CACHE_IMAGES (the RAM cap)
        # and is the main lever for avoiding a slow multi-hour re-scan of
        # Hugging Face on every epoch.
        self.disk_cache_enabled = os.getenv("DISK_CACHE", "true").lower() == "true"
        self.disk_cache_dir = Path(os.getenv("DISK_CACHE_DIR", "image_cache"))
        self._disk_chunk_files: List[Path] = []
        if self.disk_cache_enabled:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            # Resume from a previous run's disk cache if one is already there
            existing = sorted(self.disk_cache_dir.glob("chunk_*.pt"))
            if existing:
                self._disk_chunk_files = existing
                self._cache_ready = True
                log.info(f"   💽 Found {len(existing)} cached chunk files on disk "
                         f"from a previous run - will replay from disk, no HF re-stream")
        
        # Build species mapping FIRST
        self._build_species_mapping()
        
        log.info(f"📊 Species mapping built: {len(self.species_to_idx)} species")
    
    def _build_species_mapping(self):
        """Build species mapping from Hugging Face + local extras."""
        
        # Get species from local extras
        local_species = set()
        extra_path = Path(self.extra_dir)
        if extra_path.exists():
            for folder in extra_path.iterdir():
                if folder.is_dir():
                    species = folder.name.replace("_", " ").strip().lower()
                    if species:
                        local_species.add(species)
        
        # Try to get species list from Hugging Face
        hf_species = set()
        try:
            from datasets import load_dataset
            
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                
                try:
                    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            
            # Detect label column
            features = ds.features
            label_col = None
            for col in ["label", "text", "name", "species", "pokemon"]:
                if col in features:
                    label_col = col
                    break
            
            if label_col:
                # Fast path: if this is a HF ClassLabel column, the full
                # species list is already in the schema — no need to scan
                # rows at all. With 1,150 species, scanning rows to find
                # them (the old approach, capped at 500 rows) would miss
                # most species regardless of MAX_SPECIES.
                col_feature = features[label_col]
                names = getattr(col_feature, "names", None)
                if names:
                    hf_species = {str(n).strip().lower() for n in names}
                    log.info(f"   📋 Got {len(hf_species)} species directly from dataset schema")
                else:
                    # Fallback: scan rows. Raised from 500 -> 20000 with a
                    # heartbeat so it doesn't look frozen, but the schema
                    # path above should be what actually fires.
                    count = 0
                    for row in ds:
                        raw_label = row[label_col]
                        if isinstance(raw_label, int):
                            raw_label = features[label_col].int2str(raw_label)
                        species = str(raw_label).strip().lower()
                        hf_species.add(species)
                        count += 1
                        if count % 2000 == 0:
                            log.info(f"   🔎 Scanned {count} rows, found {len(hf_species)} species so far")
                        if count > 20000:
                            break
        except Exception as e:
            log.warning(f"Could not get species from Hugging Face: {e}")
        
        # Combine species
        all_species = sorted(local_species | hf_species)
        
        if not all_species:
            log.error("❌ No species found!")
            return
        
        self.species_to_idx = {s: i for i, s in enumerate(all_species)}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
        
        # Limit species if needed
        if MAX_SPECIES > 0 and len(all_species) > MAX_SPECIES:
            # Keep local species first, then add from HF
            local_list = list(local_species)
            hf_list = [s for s in all_species if s not in local_species]
            selected = local_list + hf_list[:MAX_SPECIES - len(local_list)]
            self.species_to_idx = {s: i for i, s in enumerate(selected)}
            self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
            log.info(f"   Limited to {len(selected)} species (MAX_SPECIES={MAX_SPECIES})")
    
    def __iter__(self) -> Iterator:
        """
        Epoch 1: streams local images + Hugging Face images, caching every
        yielded (tensor, label) pair. Stops hitting HF as soon as every
        species has MAX_IMAGES_PER_SPECIES images (instead of scanning the
        whole remaining dataset for nothing).
        Epoch 2+: replays from the in-memory cache, no network at all.
        """
        
        if self._disk_chunk_files:
            log.info(f"   💽 Replaying {len(self._disk_chunk_files)} chunk files from disk "
                     f"(no HF re-stream, no network)")
            for chunk_path in self._disk_chunk_files:
                try:
                    chunk = torch.load(chunk_path, weights_only=False)
                    for item in chunk:
                        yield item
                    del chunk
                except Exception as e:
                    log.warning(f"   ⚠️ Failed to load cache chunk {chunk_path}: {e}")
            gc.collect()
            return
        
        if self._cache_ready:
            log.info(f"   ♻️  Replaying {len(self._cache)} cached images (no HF re-stream)")
            for item in self._cache:
                yield item
            return
        
        target_total = len(self.species_to_idx) * MAX_IMAGES_PER_SPECIES
        species_counts: Dict[str, int] = {}
        collected = 0
        cache_capped_warned = False
        t_start = time.time()
        
        # Buffer for the disk-backed cache - flushed to a new chunk file
        # every DISK_CHUNK_SIZE images so we're never holding more than
        # one chunk's worth in RAM for this purpose.
        DISK_CHUNK_SIZE = 200
        disk_buffer: List[Tuple[torch.Tensor, int]] = []
        chunk_idx = 0
        
        def _flush_disk_chunk():
            nonlocal disk_buffer, chunk_idx
            if not self.disk_cache_enabled or not disk_buffer:
                return
            chunk_path = self.disk_cache_dir / f"chunk_{chunk_idx:05d}.pt"
            try:
                torch.save(disk_buffer, chunk_path)
                self._disk_chunk_files.append(chunk_path)
                chunk_idx += 1
            except Exception as e:
                log.warning(f"   ⚠️ Failed to write cache chunk to disk: {e}")
            disk_buffer = []
            gc.collect()
        
        def _emit(tensor, label, species):
            nonlocal collected, cache_capped_warned
            species_counts[species] = species_counts.get(species, 0) + 1
            collected += 1
            # Cap cache growth so a large run can't OOM the container.
            # Images beyond the cap still get trained on this epoch (via
            # the yield below), they just aren't retained for epoch 2+.
            #
            # BUG FIXED: `MAX_CACHE_IMAGES <= 0 or ...` used to mean "0 = no
            # limit" (the opposite of what the config comment above promises
            # - "set to 0 to disable caching entirely"). With the default of
            # 0 this appended EVERY streamed image to an unbounded in-memory
            # list - on a ~27,975-image run that's several GB of uint8
            # tensors, on top of the disk-backed chunk cache already doing
            # the real epoch-2+ replay job below. That's what was causing
            # the OOM kill. Now: when disk caching is on (the default),
            # skip the in-RAM copy entirely since disk chunks already do
            # cross-epoch replay. When disk caching is off, 0 actually
            # disables the RAM cache as documented; a positive value caps it.
            if self.disk_cache_enabled:
                pass
            elif MAX_CACHE_IMAGES > 0 and len(self._cache) < MAX_CACHE_IMAGES:
                self._cache.append((tensor, label))
            elif MAX_CACHE_IMAGES > 0 and not cache_capped_warned:
                cache_capped_warned = True
                log.warning(f"   ⚠️ Cache cap ({MAX_CACHE_IMAGES} images) reached - "
                            f"later epochs will replay only the cached subset, "
                            f"not the full {target_total}-image target. Raise "
                            f"MAX_CACHE_IMAGES if you have RAM to spare.")
            if self.disk_cache_enabled:
                disk_buffer.append((tensor, label))
                if len(disk_buffer) >= DISK_CHUNK_SIZE:
                    _flush_disk_chunk()
            return tensor, label
        
        # First, yield local images
        extra_path = Path(self.extra_dir)
        if extra_path.exists():
            for folder in extra_path.iterdir():
                if not folder.is_dir():
                    continue
                
                species = folder.name.replace("_", " ").strip().lower()
                if species not in self.species_to_idx:
                    continue
                
                label = self.species_to_idx[species]
                valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
                
                for img_path in folder.iterdir():
                    if species_counts.get(species, 0) >= MAX_IMAGES_PER_SPECIES:
                        break
                    if img_path.suffix.lower() not in valid_extensions:
                        continue
                    
                    try:
                        img = Image.open(img_path).convert("RGB")
                        if img.size[0] > 10 and img.size[1] > 10:
                            yield _emit(self.transform(img), label, species)
                    except Exception:
                        continue
        
        log.info(f"   📂 Local images collected: {collected}/{target_total}")
        
        if collected >= target_total:
            _flush_disk_chunk()
            self._cache_ready = True
            log.info(f"   ✅ Quota already met from local images, skipping HF stream")
            return
        
        # Then, stream from Hugging Face
        try:
            from datasets import load_dataset
            
            ds = load_dataset(DATASET_NAME, split="train", streaming=True)
            
            # Detect columns
            features = ds.features
            label_col = None
            image_col = None
            
            for col in ["label", "text", "name", "species", "pokemon"]:
                if col in features:
                    label_col = col
                    break
            for col in ["image", "img", "picture"]:
                if col in features:
                    image_col = col
                    break
            
            if not label_col or not image_col:
                label_col = list(features.keys())[0]
                image_col = list(features.keys())[1] if len(features) > 1 else list(features.keys())[0]
            
            rows_scanned = 0
            
            for row in ds:
                rows_scanned += 1
                
                # Heartbeat so it never looks frozen, even mid-scan
                if rows_scanned % 200 == 0:
                    elapsed = time.time() - t_start
                    log.info(f"   🔎 Scanned {rows_scanned} HF rows, "
                             f"kept {collected}/{target_total} images "
                             f"({elapsed:.0f}s elapsed)")
                
                try:
                    raw_label = row[label_col]
                    if isinstance(raw_label, int):
                        raw_label = features[label_col].int2str(raw_label)
                    
                    species = str(raw_label).strip().lower()
                    
                    if species not in self.species_to_idx:
                        continue
                    
                    if species_counts.get(species, 0) >= MAX_IMAGES_PER_SPECIES:
                        continue
                    
                    img = row[image_col]
                    if img is None:
                        continue
                    
                    if not isinstance(img, Image.Image):
                        img = Image.open(BytesIO(img))
                    
                    if img.size[0] < 10 or img.size[1] < 10:
                        continue
                    
                    label = self.species_to_idx[species]
                    
                    yield _emit(self.transform(img), label, species)
                    
                    # Stop as soon as every species has its quota instead
                    # of scanning the rest of the dataset for nothing.
                    if collected >= target_total:
                        log.info(f"   ✅ Quota met ({collected}/{target_total}) "
                                 f"after scanning {rows_scanned} HF rows")
                        break
                    
                except Exception:
                    continue
                
        except Exception as e:
            log.warning(f"Error streaming from Hugging Face: {e}")
        
        if collected < target_total:
            log.warning(f"   ⚠️ Only found {collected}/{target_total} images "
                        f"before HF dataset was exhausted")
        
        _flush_disk_chunk()
        self._cache_ready = True
    
    def get_num_species(self) -> int:
        return len(self.species_to_idx)


# ============ STREAMING TRAINER ============

def store_chunk_features_to_db(features_tensor, batch_labels, idx_to_species: Dict[int, str], db):
    """
    Saves the features for a chunk that was JUST trained on, straight to
    the DB, using the features already computed during that chunk's
    forward pass (no extra model call needed). Called right after every
    small chunk so results are persisted and the chunk can be dropped
    from memory immediately - nothing waits until end-of-epoch.
    """
    feature_buffer: Dict[str, List[np.ndarray]] = {}
    feats_np = features_tensor.detach().cpu().numpy()
    labels_list = batch_labels.detach().cpu().tolist()
    for feat, lbl in zip(feats_np, labels_list):
        if not np.all(feat == 0):
            species = idx_to_species.get(lbl)
            if species:
                feature_buffer.setdefault(species, []).append(feat)
    # One cursor, one commit for the whole chunk - see add_pokemon_features_batch.
    db.add_pokemon_features_batch(feature_buffer)


def store_features_to_db(model, dataset, db, label="checkpoint"):
    """
    Batched feature extraction + incremental DB write, using extract_batch
    (not one-image-at-a-time) so it's fast even at thousands of images.
    Called after every improved checkpoint, not just once at the very end,
    so a crash mid-training doesn't lose everything collected so far.
    """
    log.info(f"\n💾 Storing features to database ({label})...")
    t_start = time.time()
    
    image_buffer = []
    label_buffer = []
    feature_buffer: Dict[str, List[np.ndarray]] = {}
    count = 0
    
    def flush_batch():
        nonlocal count
        if not image_buffer:
            return
        pil_imgs = [transforms.ToPILImage()(img.cpu()) for img in image_buffer]
        features = model.feature_extractor.extract_batch(pil_imgs)
        for feat, lbl in zip(features, label_buffer):
            if not np.all(feat == 0):
                species = dataset.idx_to_species[lbl]
                feature_buffer.setdefault(species, []).append(feat)
                count += 1
        image_buffer.clear()
        label_buffer.clear()
    
    for img, label in dataset:
        image_buffer.append(img)
        label_buffer.append(label)
        if len(image_buffer) >= 50:
            flush_batch()
            db.add_pokemon_features_batch(feature_buffer)
            feature_buffer = {}
            gc.collect()
    
    flush_batch()
    db.add_pokemon_features_batch(feature_buffer)
    
    elapsed = time.time() - t_start
    log.info(f"   ✅ Stored {count} features in {elapsed:.0f}s")


def stream_train():    
    # ============ PRE-LOAD MODEL (NO DOWNLOAD DURING TRAINING) ============

    log.info("📥 Pre-loading AI model...")
    try:
        from torchvision import models
        _ = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        log.info("✅ Model loaded and cached!")
    except Exception as e:
        log.warning(f"⚠️ Model pre-load failed: {e}")
    
    log_memory("baseline, right after model load")
    
    """Train using streaming - process in batches, clear memory."""
    
    log.info("")
    log.info("🚀 Pokémon AI Trainer - STREAMING MODE")
    log.info("=" * 60)
    log.info("   ✅ Processes images in batches")
    log.info("   ✅ Clears memory after each batch")
    log.info("   ✅ Saves to database incrementally")
    log.info("=" * 60)
    
    if HF_TOKEN:
        log.info(f"🔑 HF_TOKEN: ✅ Set")
    else:
        log.warning(f"🔑 HF_TOKEN: ❌ Not set")
    
    log.info("\n📦 Checking for archive files...")
    extract_archive_files()
    
    log.info("\n📂 Connecting to database...")
    db = Database()
    stats = db.get_stats()
    log.info(f"   Existing species: {stats['total_species']}")
    log.info(f"   Existing features: {stats['total_features']}")
    
    # Create streaming dataset
    log.info("\n📂 Initializing streaming dataset...")
    dataset = StreamingPokemonDataset(extra_dir="Extra pokemons")
    num_species = dataset.get_num_species()
    
    if num_species == 0:
        log.error("❌ No species found! Exiting.")
        return
    
    log.info(f"   📊 Total species: {num_species}")
    
    # Initialize model
    log.info("\n🧠 Initializing model...")
    model = PokemonClassifier(num_species=num_species)
    model.to(DEVICE)
    # Backbone (shufflenet) stays frozen - extract_batch always runs it under
    # torch.no_grad() regardless of this train()/eval() call. `projection`
    # has no BatchNorm/Dropout so train()/eval() doesn't change its math -
    # this call just needs to not be .eval() so nothing here is confusing
    # later. What actually makes projection trainable is passing its
    # parameters to the optimizer below + forward_batch(..., grad=True).
    model.feature_extractor.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(model.classifier.parameters()) + list(model.feature_extractor.projection.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    log.info(f"\n🎯 Training for {EPOCHS} epochs...")
    log.info(f"   Species: {num_species}")
    log.info(f"   Batch size: {BATCH_SIZE}")
    log.info(f"   Stream batch: {STREAM_BATCH_SIZE} images per batch")
    log.info(f"   Learning rate: {LEARNING_RATE}")
    log.info(f"   Early-stop target: {EARLY_STOP_ACC}% train acc, "
             f"or {EARLY_STOP_PATIENCE} epochs without val improvement")
    log.info("-" * 60)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    stop_training = False
    start_epoch = 0

    # ============ RESUME FROM CHECKPOINT IF ONE EXISTS ============
    if os.path.exists(CHECKPOINT_PATH):
        try:
            log.info(f"\n💾 Found checkpoint at {CHECKPOINT_PATH}, resuming training...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
            log.info(f"   ✅ Resuming at epoch {start_epoch + 1}/{EPOCHS} "
                     f"(best val acc so far: {best_val_acc:.1f}%)")
        except Exception as e:
            log.warning(f"   ⚠️ Failed to load checkpoint ({e}), starting fresh instead")
            start_epoch = 0
            best_val_acc = 0.0
            epochs_without_improvement = 0
    else:
        log.info(f"\n💾 No checkpoint found at {CHECKPOINT_PATH}, starting fresh")

    if start_epoch >= EPOCHS:
        log.info(f"   ℹ️ Checkpoint already completed all {EPOCHS} epochs - nothing to resume")

    for epoch in range(start_epoch, EPOCHS):
        log.info(f"\n📊 Epoch {epoch+1}/{EPOCHS}")
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        # Stream images
        image_buffer = []
        label_buffer = []
        
        for img, label in PrefetchIterator(dataset, maxsize=STREAM_BATCH_SIZE * 2):
            image_buffer.append(img)
            label_buffer.append(label)
            
            # When buffer reaches STREAM_BATCH_SIZE, process it
            if len(image_buffer) >= STREAM_BATCH_SIZE:
                # Create a mini-batch from buffer
                batch_imgs = torch.stack(image_buffer)
                batch_labels = torch.tensor(label_buffer)
                
                # Move to device
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Convert to PIL for feature extraction. Cached tensors are
                # raw uint8 (see StreamingPokemonDataset.transform) so no
                # un-normalize step is needed - ToPILImage handles uint8
                # CHW tensors directly.
                pil_imgs = []
                for i in range(batch_imgs.size(0)):
                    try:
                        pil_img = transforms.ToPILImage()(batch_imgs[i].cpu())
                        if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                            pil_imgs.append(pil_img)
                    except Exception:
                        continue
                
                if not pil_imgs:
                    # Clear buffer and continue
                    image_buffer = []
                    label_buffer = []
                    continue
                
                features, logits = None, None
                try:
                    features, logits = model.forward_batch(pil_imgs, grad=True)
                    features = features.to(DEVICE)
                    logits = logits.to(DEVICE)
                    
                    loss = criterion(logits, batch_labels[:len(pil_imgs)])
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch_labels[:len(pil_imgs)].size(0)
                    _, predicted = torch.max(logits, 1)
                    train_total += batch_labels[:len(pil_imgs)].size(0)
                    train_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
                    batch_count += 1
                    
                    # Save this chunk's results to the DB right away, then
                    # it's safe to drop from memory below - nothing about
                    # this chunk needs to be revisited later.
                    store_chunk_features_to_db(features, batch_labels[:len(pil_imgs)],
                                                dataset.idx_to_species, db)
                    
                    # Progress update (every batch — cheap now that images are cached)
                    acc = 100 * train_correct / train_total if train_total > 0 else 0
                    log.info(f"   📊 Batch {batch_count}: Loss: {train_loss/train_total:.3f}, Acc: {acc:.1f}%")
                    
                    # Stop this epoch early once training accuracy hits the
                    # target — no point grinding through remaining batches.
                    # Require a few batches first so one lucky batch doesn't
                    # trigger a false stop.
                    if batch_count >= 3 and acc >= EARLY_STOP_ACC:
                        log.info(f"   🎯 Reached target accuracy ({acc:.1f}% >= "
                                 f"{EARLY_STOP_ACC}%), moving on early")
                        del batch_imgs, batch_labels, pil_imgs
                        gc.collect()
                        break
                    
                except Exception as e:
                    log.warning(f"   ⚠️ Batch failed: {e}")
                
                # CLEAR MEMORY!
                del batch_imgs
                del batch_labels
                del pil_imgs
                del features, logits
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                log_memory(f"batch {batch_count}")
                
                # Reset buffer
                image_buffer = []
                label_buffer = []
        
        # Process any remaining images (skip if we already hit target accuracy early)
        if image_buffer and not (batch_count >= 3 and (100 * train_correct / train_total if train_total > 0 else 0) >= EARLY_STOP_ACC):
            batch_imgs = torch.stack(image_buffer)
            batch_labels = torch.tensor(label_buffer)
            
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            pil_imgs = []
            for i in range(batch_imgs.size(0)):
                try:
                    pil_img = transforms.ToPILImage()(batch_imgs[i].cpu())
                    if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                        pil_imgs.append(pil_img)
                except Exception:
                    continue
            
            if pil_imgs:
                try:
                    features, logits = model.forward_batch(pil_imgs, grad=True)
                    features = features.to(DEVICE)
                    logits = logits.to(DEVICE)
                    
                    loss = criterion(logits, batch_labels[:len(pil_imgs)])
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch_labels[:len(pil_imgs)].size(0)
                    _, predicted = torch.max(logits, 1)
                    train_total += batch_labels[:len(pil_imgs)].size(0)
                    train_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
                    batch_count += 1
                    
                    # Same as the main loop: persist this last chunk's
                    # results immediately rather than waiting for an
                    # end-of-epoch save.
                    store_chunk_features_to_db(features, batch_labels[:len(pil_imgs)],
                                                dataset.idx_to_species, db)
                    
                except Exception:
                    pass
                
                del batch_imgs
                del batch_labels
                del pil_imgs
                gc.collect()
                log_memory("leftover chunk")
        
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
        # Validation - sample validation (use last batch as validation)
        model.eval()
        val_correct = 0
        val_total = 0
        
        # Use a small validation set (last few images)
        val_images = []
        val_labels = []
        val_count = 0
        
        for img, label in dataset:
            val_images.append(img)
            val_labels.append(label)
            val_count += 1
            if val_count >= 50:  # Small validation set
                break
        
        if val_images:
            batch_imgs = torch.stack(val_images).to(DEVICE)
            batch_labels = torch.tensor(val_labels).to(DEVICE)
            
            pil_imgs = []
            for i in range(batch_imgs.size(0)):
                try:
                    pil_img = transforms.ToPILImage()(batch_imgs[i].cpu())
                    if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                        pil_imgs.append(pil_img)
                except Exception:
                    continue
            
            if pil_imgs:
                with torch.no_grad():
                    _, logits = model.forward_batch(pil_imgs)
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_labels[:len(pil_imgs)].size(0)
                    val_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
            
            del batch_imgs
            del batch_labels
            del pil_imgs
            gc.collect()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        log.info(f"\n📊 Epoch {epoch+1} Results:")
        log.info(f"   Train Acc: {train_acc:.1f}%")
        log.info(f"   Val Acc: {val_acc:.1f}%")
        log.info(f"   Total batches: {batch_count}")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(MODEL_OUTPUT) or ".", exist_ok=True)
            torch.save(model.feature_extractor.state_dict(), MODEL_OUTPUT)
            log.info(f"   ✅ Saved model (val acc: {val_acc:.1f}%)")
            # Write features to DB right away too — don't wait until the
            # whole run finishes. If Railway crashes on a later epoch,
            # this checkpoint's data is already safe in the DB.
            store_features_to_db(model, dataset, db, label=f"epoch {epoch+1}, val acc {val_acc:.1f}%")
        else:
            epochs_without_improvement += 1

        # Full training-state checkpoint, written EVERY epoch (regardless of
        # whether val acc improved) - this is what makes a restart resume
        # instead of starting over. Includes model weights (classifier +
        # feature_extractor, so the projection layer's learned weights
        # aren't lost either), optimizer state (AdamW's per-parameter
        # moment estimates - skipping this would effectively reset the
        # optimizer and cause a training hiccup right after every resume),
        # scheduler state, and the epoch/best-acc bookkeeping needed to
        # pick up counting from the right place.
        try:
            os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
            }, CHECKPOINT_PATH)
            log.info(f"   💾 Checkpoint saved (epoch {epoch+1}/{EPOCHS})")
        except Exception as e:
            log.warning(f"   ⚠️ Failed to save checkpoint: {e}")

        # Stop across epochs if we've hit the target or plateaued
        if train_acc >= EARLY_STOP_ACC:
            log.info(f"   🎯 Training accuracy target reached ({train_acc:.1f}% >= "
                     f"{EARLY_STOP_ACC}%), stopping training")
            stop_training = True
        elif epochs_without_improvement >= EARLY_STOP_PATIENCE:
            log.info(f"   ⏸️ Val accuracy hasn't improved in {EARLY_STOP_PATIENCE} "
                     f"epochs (best: {best_val_acc:.1f}%), stopping early")
            stop_training = True
        
        if stop_training:
            break
    
    log.info("-" * 60)
    log.info(f"✅ Training complete!")
    log.info(f"   Best validation accuracy: {best_val_acc:.1f}%")
    log.info(f"   Model saved to: {MODEL_OUTPUT}")
    
    final_stats = db.get_stats()
    log.info(f"\n📊 Database Stats:")
    log.info(f"   Total species: {final_stats['total_species']}")
    log.info(f"   Total features: {final_stats['total_features']}")
    log.info(f"   Using Turso: {final_stats['use_turso']}")
    
    log.info("\n" + "=" * 60)
    log.info("✅ All done! Model is ready to use.")
    log.info("=" * 60)
    
    db.close()


if __name__ == "__main__":
    stream_train()
