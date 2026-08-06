"""
train_model.py - Complete AI Pokémon Trainer for Railway
Supports: ZIP, RAR (with automatic unrar fallback)
Fetches from Hugging Face (with HF_TOKEN) + your extra Pokémon
Trains model, stores in database
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO

# ============ COMPLETE LOG SUPPRESSION ============
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("trainer")

# Suppress external loggers
for logger_name in [
    "urllib3", "requests", "datasets", "huggingface_hub",
    "filelock", "transformers", "torch", "torchvision",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).disabled = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm

# ============ CONFIGURATION ============

TURSO_URL = os.getenv("TURSO_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_IMAGES_PER_SPECIES = int(os.getenv("MAX_IMAGES_PER_SPECIES", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
DATASET_NAME = os.getenv("DATASET_NAME", "SpreadSheets/Poketwo-Spawn-Images")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
DB_PATH = os.getenv("DB_PATH", "pokemon.db")
AUTO_EXTRACT_ARCHIVES = os.getenv("AUTO_EXTRACT_ARCHIVES", "true").lower() == "true"

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
        """Extract feature vector from a single image."""
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
        except Exception as e:
            log.warning(f"Feature extraction failed: {e}")
            return np.zeros(256)
    
    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Extract features from multiple images."""
        valid_images = []
        for img in images:
            if img is not None and isinstance(img, Image.Image):
                try:
                    valid_images.append(self.transform(img).unsqueeze(0))
                except Exception:
                    continue
        
        if not valid_images:
            log.warning("No valid images in batch")
            return np.zeros((len(images), 256))
        
        try:
            batch_tensor = torch.cat(valid_images, dim=0).to(DEVICE)
            batch_tensor = self.normalize(batch_tensor)
            features = self.backbone(batch_tensor)
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)
            
            # If we dropped some images, pad the result
            result = projected.cpu().numpy()
            if len(valid_images) < len(images):
                padded = np.zeros((len(images), result.shape[1]))
                padded[:len(valid_images)] = result
                return padded
            return result
        except Exception as e:
            log.warning(f"Batch feature extraction failed: {e}")
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


# ============ DATASET ============

class PokemonDataset(Dataset):
    def __init__(self, species_to_idx: Dict[str, int], max_per_species: int = 10, extra_dir: str = "Extra pokemons"):
        self.species_to_idx = species_to_idx
        self.max_per_species = max_per_species
        self.samples = []
        self.species_stats = {}
        self.species_images = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load from Hugging Face
        self._load_huggingface()
        
        # Load local extras
        self._load_extra_pokemon(extra_dir)
        
        random.shuffle(self.samples)
        self._print_stats()
    
    def _load_huggingface(self):
        log.info(f"📥 Loading Hugging Face dataset...")
        
        try:
            from datasets import load_dataset
            
            import warnings
            warnings.filterwarnings("ignore")
            logging.getLogger("datasets").setLevel(logging.ERROR)
            
            ds = load_dataset(DATASET_NAME, split="train", streaming=True)
            
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
            
            species_set = set(self.species_to_idx.keys())
            species_counts = {}
            count = 0
            
            log.info(f"   🔄 Scanning dataset for matching species...")
            
            for row in ds:
                try:
                    raw_label = row[label_col]
                    if isinstance(raw_label, int):
                        raw_label = features[label_col].int2str(raw_label)
                    
                    species = str(raw_label).strip().lower()
                    
                    if species not in species_set:
                        continue
                    if species_counts.get(species, 0) >= self.max_per_species:
                        continue
                    
                    img = row[image_col]
                    if img is None:
                        continue
                    
                    if not isinstance(img, Image.Image):
                        img = Image.open(BytesIO(img))
                    
                    # Verify image is valid
                    if img.size[0] < 10 or img.size[1] < 10:
                        continue
                    
                    self.samples.append((img, self.species_to_idx[species]))
                    self.species_images.setdefault(species, []).append(img)
                    species_counts[species] = species_counts.get(species, 0) + 1
                    count += 1
                    
                    if count % 100 == 0:
                        log.info(f"   📊 Checkpoint: {count} images loaded from Hugging Face")
                    
                except Exception as e:
                    continue
                
                if count > 1000:
                    break
            
            self.species_stats.update(species_counts)
            log.info(f"   ✅ Loaded {len(self.samples)} samples from Hugging Face")
            return True
            
        except Exception as e:
            log.warning(f"   ⚠️ Failed to load Hugging Face dataset: {e}")
            return False
    
    def _load_extra_pokemon(self, extra_dir: str):
        extra_path = Path(extra_dir)
        if not extra_path.exists():
            return
        
        log.info(f"📁 Loading extra Pokémon from: {extra_dir}")
        
        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"}
        species_counts = {}
        
        for folder in extra_path.iterdir():
            if not folder.is_dir():
                continue
            
            species = folder.name.replace("_", " ").strip().lower()
            if not species:
                species = "unknown"
            
            if species not in self.species_to_idx:
                idx = len(self.species_to_idx)
                self.species_to_idx[species] = idx
            
            images = []
            for ext in valid_extensions:
                images.extend(folder.glob(f"*{ext}"))
            
            if not images:
                continue
            
            current_count = species_counts.get(species, 0)
            available = self.max_per_species - current_count
            if available <= 0:
                continue
            
            images = images[:available]
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    if img.size[0] < 10 or img.size[1] < 10:
                        continue
                    self.samples.append((img, self.species_to_idx[species]))
                    self.species_images.setdefault(species, []).append(img)
                    species_counts[species] = species_counts.get(species, 0) + 1
                except Exception:
                    continue
        
        for species, count in species_counts.items():
            self.species_stats[species] = self.species_stats.get(species, 0) + count
        
        log.info(f"   ✅ Loaded {len(self.samples)} samples from extra Pokémon")
    
    def _print_stats(self):
        log.info("=" * 60)
        log.info("📊 DATASET STATISTICS")
        log.info("=" * 60)
        log.info(f"   Total samples: {len(self.samples)}")
        log.info(f"   Total species: {len(self.species_stats)}")
        
        if self.species_stats:
            counts = list(self.species_stats.values())
            log.info(f"   Avg per species: {sum(counts) / len(counts):.1f}")
            log.info(f"   Min per species: {min(counts)}")
            log.info(f"   Max per species: {max(counts)}")
        
        low_species = [s for s, c in self.species_stats.items() if c < 3]
        if low_species:
            log.warning(f"\n⚠️ {len(low_species)} species have < 3 images:")
            for s in low_species[:5]:
                log.warning(f"   - {s}: {self.species_stats[s]} images")
            if len(low_species) > 5:
                log.warning(f"   ... and {len(low_species) - 5} more")
        
        if len(self.samples) < 10:
            log.error("❌ Too few samples! Need at least 10 images to train.")
        
        log.info("=" * 60)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if isinstance(img, Image.Image):
            img = self.transform(img)
        return img, label


# ============ TRAINING ============

def train():
    log.info("")
    log.info("🚀 Pokémon AI Trainer")
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
    
    extra_path = Path("Extra pokemons")
    extra_species = []
    if extra_path.exists():
        extra_species = [f.name.replace("_", " ").strip().lower() 
                         for f in extra_path.iterdir() if f.is_dir()]
        log.info(f"📁 Found {len(extra_species)} extra Pokémon species")
        
        for s in extra_species[:5]:
            folder = extra_path / s.replace(" ", "_")
            count = len(list(folder.glob("*"))) if folder.exists() else 0
            log.info(f"      - {s}: {count} images")
        if len(extra_species) > 5:
            log.info(f"      ... and {len(extra_species) - 5} more")
    
    if not extra_species:
        log.error("❌ No extra Pokémon found! Please upload images.")
        return
    
    all_species = sorted(extra_species)
    log.info(f"   Total species: {len(all_species)}")
    
    species_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    
    log.info("\n📂 Creating dataset...")
    dataset = PokemonDataset(
        species_to_idx=species_to_idx,
        max_per_species=MAX_IMAGES_PER_SPECIES,
        extra_dir="Extra pokemons"
    )
    
    if len(dataset) == 0:
        log.error("❌ No samples loaded! Exiting.")
        return
    
    if len(dataset) < 10:
        log.error("❌ Too few samples! Need at least 10 images to train.")
        return
    
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, train_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, val_size), shuffle=False, num_workers=0)
    
    log.info(f"\n📊 Dataset split:")
    log.info(f"   Train samples: {train_size}")
    log.info(f"   Val samples: {val_size}")
    
    log.info("\n🧠 Initializing model...")
    model = PokemonClassifier(num_species=len(all_species))
    model.to(DEVICE)
    model.feature_extractor.eval()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    log.info(f"\n🎯 Training for {EPOCHS} epochs...")
    log.info(f"   Batch size: {min(BATCH_SIZE, train_size)}")
    log.info(f"   Learning rate: {LEARNING_RATE}")
    log.info("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_imgs, batch_labels in progress_bar:
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            pil_imgs = []
            for i in range(batch_imgs.size(0)):
                try:
                    img = batch_imgs[i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1)
                    pil_img = transforms.ToPILImage()(img)
                    if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                        pil_imgs.append(pil_img)
                except Exception as e:
                    continue
            
            if not pil_imgs:
                log.warning("No valid images in batch, skipping...")
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
                
                progress_bar.set_postfix({
                    "loss": f"{train_loss/train_total:.3f}" if train_total > 0 else "0.000",
                    "acc": f"{100*train_correct/train_total:.1f}%" if train_total > 0 else "0.0%"
                })
            except Exception as e:
                log.warning(f"Batch training failed: {e}")
                continue
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                pil_imgs = []
                for i in range(batch_imgs.size(0)):
                    try:
                        img = batch_imgs[i].cpu()
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img = img * std + mean
                        img = torch.clamp(img, 0, 1)
                        pil_img = transforms.ToPILImage()(img)
                        if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                            pil_imgs.append(pil_img)
                    except Exception:
                        continue
                
                if not pil_imgs:
                    continue
                
                try:
                    _, logits = model.forward_batch(pil_imgs)
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_labels[:len(pil_imgs)].size(0)
                    val_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
                except Exception:
                    continue
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
        log.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        
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
    
    log.info("\n💾 Storing features in database...")
    
    species_images = {}
    for img, label_idx in dataset.samples:
        species = idx_to_species[label_idx]
        species_images.setdefault(species, []).append(img)
    
    log.info(f"   Extracting features for {len(species_images)} species...")
    
    for species, images in tqdm(species_images.items(), desc="   Storing"):
        try:
            features = []
            for img in images:
                if isinstance(img, Image.Image) and img.size[0] > 10 and img.size[1] > 10:
                    feature = model.feature_extractor.extract(img)
                    if not np.all(feature == 0):
                        features.append(feature)
            if features:
                db.add_pokemon_features(species, features)
        except Exception as e:
            log.error(f"   Failed to store {species}: {e}")
    
    info_path = os.path.join(os.path.dirname(MODEL_OUTPUT) or ".", "species_info.json")
    with open(info_path, 'w') as f:
        json.dump({
            'species_list': all_species,
            'num_species': len(all_species),
            'best_val_acc': best_val_acc,
            'max_per_species': MAX_IMAGES_PER_SPECIES,
            'species_stats': dataset.species_stats,
            'extra_pokemon': extra_species
        }, f, indent=2)
    log.info(f"   Species info saved to: {info_path}")
    
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
    train()
