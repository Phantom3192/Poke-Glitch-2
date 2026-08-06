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

# ============ CONFIGURATION ============

TURSO_URL = os.getenv("TURSO_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Increased for streaming
STREAM_BATCH_SIZE = int(os.getenv("STREAM_BATCH_SIZE", "100"))  # Images per stream batch
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
DATASET_NAME = os.getenv("DATASET_NAME", "SpreadSheets/Poketwo-Spawn-Images")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
DB_PATH = os.getenv("DB_PATH", "pokemon.db")
AUTO_EXTRACT_ARCHIVES = os.getenv("AUTO_EXTRACT_ARCHIVES", "true").lower() == "true"
MAX_SPECIES = int(os.getenv("MAX_SPECIES", "100"))  # Max species to train
MAX_IMAGES_PER_SPECIES = int(os.getenv("MAX_IMAGES_PER_SPECIES", "10"))

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
        if variant_names is None:
            variant_names = [f"{species}_{i+1}" for i in range(len(features))]
        
        cursor = self._conn.cursor()
        for i, feature in enumerate(features):
            feature_json = json.dumps(feature.tolist())
            variant_name = variant_names[i] if i < len(variant_names) else f"{species}_{i+1}"
            cursor.execute("""
                INSERT OR REPLACE INTO pokemon_features 
                (species, variant_name, feature_vector, created_at)
                VALUES (?, ?, ?, strftime('%s', 'now'))
            """, (species, variant_name, feature_json))
        
        cursor.execute("""
            INSERT OR REPLACE INTO species_info (species, count, last_updated)
            VALUES (?, ?, strftime('%s', 'now'))
        """, (species, len(features)))
        self._conn.commit()
    
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
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        backbone_dim = 1280
        
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
            features = self.backbone(img_tensor)
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)
            return projected.cpu().numpy().flatten()
        except Exception:
            return np.zeros(256)
    
    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
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
            features = self.backbone(batch_tensor)
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)
            
            result = projected.cpu().numpy()
            if len(valid_images) < len(images):
                padded = np.zeros((len(images), result.shape[1]))
                padded[:len(valid_images)] = result
                return padded
            return result
        except Exception:
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
    
    def forward_batch(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor.extract_batch(images)
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
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
                count = 0
                for row in ds:
                    raw_label = row[label_col]
                    if isinstance(raw_label, int):
                        raw_label = features[label_col].int2str(raw_label)
                    species = str(raw_label).strip().lower()
                    hf_species.add(species)
                    count += 1
                    if count > 500:  # Get enough for mapping
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
        """Stream images one by one from Hugging Face."""
        
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
                    if img_path.suffix.lower() not in valid_extensions:
                        continue
                    
                    try:
                        img = Image.open(img_path).convert("RGB")
                        if img.size[0] > 10 and img.size[1] > 10:
                            yield self.transform(img), label
                    except Exception:
                        continue
        
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
            
            species_counts = {}
            
            for row in ds:
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
                    species_counts[species] = species_counts.get(species, 0) + 1
                    
                    yield self.transform(img), label
                    
                except Exception:
                    continue
                
        except Exception as e:
            log.warning(f"Error streaming from Hugging Face: {e}")
            return
    
    def get_num_species(self) -> int:
        return len(self.species_to_idx)


# ============ STREAMING TRAINER ============

def stream_train():
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
    model.feature_extractor.eval()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    log.info(f"\n🎯 Training for {EPOCHS} epochs...")
    log.info(f"   Species: {num_species}")
    log.info(f"   Batch size: {BATCH_SIZE}")
    log.info(f"   Stream batch: {STREAM_BATCH_SIZE} images per batch")
    log.info(f"   Learning rate: {LEARNING_RATE}")
    log.info("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        log.info(f"\n📊 Epoch {epoch+1}/{EPOCHS}")
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        # Stream images
        image_buffer = []
        label_buffer = []
        
        for img, label in dataset:
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
                
                # Convert to PIL for feature extraction
                pil_imgs = []
                for i in range(batch_imgs.size(0)):
                    try:
                        img_tensor = batch_imgs[i].cpu()
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        pil_img = transforms.ToPILImage()(img_tensor)
                        if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                            pil_imgs.append(pil_img)
                    except Exception:
                        continue
                
                if not pil_imgs:
                    # Clear buffer and continue
                    image_buffer = []
                    label_buffer = []
                    continue
                
                try:
                    features, logits = model.forward_batch(pil_imgs)
                    features = features.to(DEVICE)
                    logits = logits.to(DEVICE)
                    
                    loss = criterion(logits, batch_labels[:len(pil_imgs)])
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    train_total += batch_labels[:len(pil_imgs)].size(0)
                    train_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
                    batch_count += 1
                    
                    # Progress update
                    if batch_count % 10 == 0:
                        acc = 100 * train_correct / train_total if train_total > 0 else 0
                        log.info(f"   📊 Batch {batch_count}: Loss: {train_loss/train_total:.3f}, Acc: {acc:.1f}%")
                    
                except Exception as e:
                    log.warning(f"   ⚠️ Batch failed: {e}")
                
                # CLEAR MEMORY!
                del batch_imgs
                del batch_labels
                del pil_imgs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
                # Reset buffer
                image_buffer = []
                label_buffer = []
        
        # Process any remaining images
        if image_buffer:
            batch_imgs = torch.stack(image_buffer)
            batch_labels = torch.tensor(label_buffer)
            
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            pil_imgs = []
            for i in range(batch_imgs.size(0)):
                try:
                    img_tensor = batch_imgs[i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    pil_img = transforms.ToPILImage()(img_tensor)
                    if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                        pil_imgs.append(pil_img)
                except Exception:
                    continue
            
            if pil_imgs:
                try:
                    features, logits = model.forward_batch(pil_imgs)
                    features = features.to(DEVICE)
                    logits = logits.to(DEVICE)
                    
                    loss = criterion(logits, batch_labels[:len(pil_imgs)])
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    train_total += batch_labels[:len(pil_imgs)].size(0)
                    train_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
                    batch_count += 1
                    
                except Exception:
                    pass
                
                del batch_imgs
                del batch_labels
                del pil_imgs
                gc.collect()
        
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
                    img_tensor = batch_imgs[i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    pil_img = transforms.ToPILImage()(img_tensor)
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
            os.makedirs(os.path.dirname(MODEL_OUTPUT) or ".", exist_ok=True)
            torch.save(model.feature_extractor.state_dict(), MODEL_OUTPUT)
            log.info(f"   ✅ Saved model (val acc: {val_acc:.1f}%)")
    
    log.info("-" * 60)
    log.info(f"✅ Training complete!")
    log.info(f"   Best validation accuracy: {best_val_acc:.1f}%")
    log.info(f"   Model saved to: {MODEL_OUTPUT}")
    
    # Store features in database
    log.info("\n💾 Storing features in database...")
    
    # Stream through dataset one more time to extract features
    feature_buffer = {}
    count = 0
    
    for img, label in dataset:
        species = dataset.idx_to_species[label]
        feature = model.feature_extractor.extract(transforms.ToPILImage()(img.cpu()))
        if not np.all(feature == 0):
            feature_buffer.setdefault(species, []).append(feature)
            count += 1
        
        # Store every 50 images to free memory
        if count % 50 == 0:
            for species, features in feature_buffer.items():
                if features:
                    db.add_pokemon_features(species, features)
            feature_buffer = {}
            gc.collect()
            log.info(f"   📊 Stored {count} features so far")
    
    # Store remaining
    for species, features in feature_buffer.items():
        if features:
            db.add_pokemon_features(species, features)
    
    log.info(f"   ✅ Stored {count} features in database")
    
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
