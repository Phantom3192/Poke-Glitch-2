"""
train_model.py - Complete AI Pokémon Trainer for Railway
Supports: ZIP, RAR (with automatic unrar fallback)
Fetches from Hugging Face + your extra Pokémon
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

# Environment variables (Railway will set these)
TURSO_URL = os.getenv("TURSO_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
MAX_IMAGES_PER_SPECIES = int(os.getenv("MAX_IMAGES_PER_SPECIES", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
DATASET_NAME = os.getenv("DATASET_NAME", "SpreadSheets/Poketwo-Spawn-Images")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
DB_PATH = os.getenv("DB_PATH", "pokemon.db")
AUTO_EXTRACT_ARCHIVES = os.getenv("AUTO_EXTRACT_ARCHIVES", "true").lower() == "true"

# Device
DEVICE = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("trainer")

# ============ ARCHIVE EXTRACTION ============

def extract_archive_files():
    """
    Extract ZIP and RAR archive files.
    Tries multiple methods for RAR extraction.
    """
    if not AUTO_EXTRACT_ARCHIVES:
        return
    
    # Find archives
    archives = []
    
    # ZIP files
    for pattern in ["*.zip", "*.ZIP"]:
        archives.extend(Path(".").glob(pattern))
    
    # RAR files
    for pattern in ["*.rar", "*.RAR"]:
        archives.extend(Path(".").glob(pattern))
    
    # 7z files
    for pattern in ["*.7z", "*.7Z"]:
        archives.extend(Path(".").glob(pattern))
    
    if not archives:
        log.info("📦 No archive files found.")
        return
    
    log.info(f"📦 Found {len(archives)} archive file(s), extracting...")
    
    extra_dir = Path("Extra pokemons")
    extra_dir.mkdir(exist_ok=True)
    
    extracted_count = 0
    species_found = set()
    
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
                extracted = extract_rar_with_fallback(archive_path, temp_dir)
                if not extracted:
                    log.error(f"   ❌ Failed to extract RAR: {archive_path.name}")
                    shutil.rmtree(temp_dir)
                    continue
            
            elif ext in ['.7z']:
                log.info(f"   📂 Extracting 7z: {archive_path.name}")
                extracted = extract_7z_with_fallback(archive_path, temp_dir)
                if not extracted:
                    log.error(f"   ❌ Failed to extract 7z: {archive_path.name}")
                    shutil.rmtree(temp_dir)
                    continue
            
            else:
                log.warning(f"   ⚠️ Unsupported archive format: {ext}")
                shutil.rmtree(temp_dir)
                continue
            
            # Process extracted files
            extracted, species = process_extracted_files(temp_dir, extra_dir)
            extracted_count += extracted
            species_found.update(species)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Remove the archive file after extraction
            archive_path.unlink()
            log.info(f"   ✅ Extracted: {archive_path.name} ({extracted} images)")
            
        except Exception as e:
            log.error(f"   ❌ Failed to extract {archive_path}: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    # Show what was extracted
    if extra_dir.exists():
        species = [f for f in extra_dir.iterdir() if f.is_dir()]
        if species:
            log.info(f"\n   📁 Extracted {len(species)} species from archives:")
            for s in species[:5]:
                count = len(list(s.glob("*")))
                log.info(f"      - {s.name}: {count} images")
            if len(species) > 5:
                log.info(f"      ... and {len(species) - 5} more")
        
        total_images = sum(len(list(f.glob("*"))) for f in species)
        log.info(f"   📊 Total images extracted: {total_images}")


def extract_rar_with_fallback(rar_path: Path, dest_dir: Path) -> bool:
    """Extract RAR using multiple methods."""
    
    # Method 1: Try rarfile library
    try:
        import rarfile
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(dest_dir)
        return True
    except ImportError:
        log.debug("   rarfile not installed, trying unrar command...")
    except Exception as e:
        log.debug(f"   rarfile failed: {e}, trying unrar command...")
    
    # Method 2: Try unrar command
    try:
        result = subprocess.run(
            ['unrar', 'x', '-y', str(rar_path), str(dest_dir)],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0:
            return True
        else:
            log.debug(f"   unrar failed: {result.stderr}")
    except FileNotFoundError:
        log.debug("   unrar command not found")
    except Exception as e:
        log.debug(f"   unrar command failed: {e}")
    
    # Method 3: Try 7z command (sometimes available)
    try:
        result = subprocess.run(
            ['7z', 'x', '-y', str(rar_path), f'-o{dest_dir}'],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0:
            return True
    except:
        pass
    
    log.error(f"   ❌ Cannot extract RAR. Please convert to ZIP format.")
    log.error(f"   Run: zip -r Extra pokemons.zip Extra pokemons/")
    return False


def extract_7z_with_fallback(archive_path: Path, dest_dir: Path) -> bool:
    """Extract 7z using multiple methods."""
    
    # Method 1: Try py7zr
    try:
        import py7zr
        with py7zr.SevenZipFile(archive_path, 'r') as sz:
            sz.extractall(dest_dir)
        return True
    except ImportError:
        log.debug("   py7zr not installed, trying 7z command...")
    except Exception as e:
        log.debug(f"   py7zr failed: {e}, trying 7z command...")
    
    # Method 2: Try 7z command
    try:
        result = subprocess.run(
            ['7z', 'x', '-y', str(archive_path), f'-o{dest_dir}'],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0:
            return True
    except:
        pass
    
    log.error(f"   ❌ Cannot extract 7z.")
    return False


def process_extracted_files(temp_dir: Path, extra_dir: Path) -> Tuple[int, set]:
    """
    Process extracted files and organize them into species folders.
    Returns (image_count, species_set).
    """
    extracted_count = 0
    species_found = set()
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', 
                        '.PNG', '.JPG', '.JPEG', '.WEBP'}
    
    # Walk through extracted files
    for root, dirs, files in os.walk(temp_dir):
        root_path = Path(root)
        
        # Check if this directory contains images
        images = [f for f in files if Path(f).suffix in valid_extensions]
        
        if images:
            # Determine species name from folder structure
            rel_path = root_path.relative_to(temp_dir)
            
            if len(rel_path.parts) == 0:
                species_name = "unknown"
            else:
                species_name = rel_path.parts[0].replace("_", " ").strip()
                if not species_name:
                    species_name = "unknown"
            
            # Create destination folder
            dest_dir = extra_dir / species_name.replace(" ", "_")
            dest_dir.mkdir(exist_ok=True)
            
            # Move images
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
                species_found.add(species_name)
        
        # Also check subdirectories
        for d in dirs:
            sub_path = root_path / d
            sub_images = [f for f in sub_path.iterdir() if f.suffix in valid_extensions]
            
            if sub_images:
                species_name = d.replace("_", " ").strip()
                if not species_name:
                    species_name = "unknown"
                
                dest_dir = extra_dir / species_name.replace(" ", "_")
                dest_dir.mkdir(exist_ok=True)
                
                for img in sub_images:
                    dest = dest_dir / img.name
                    if dest.exists():
                        counter = 1
                        stem = img.stem
                        suffix = img.suffix
                        while dest.exists():
                            dest = dest_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    shutil.move(str(img), str(dest))
                    extracted_count += 1
                    species_found.add(species_name)
    
    return extracted_count, species_found

# ============ DATABASE LAYER ============

class Database:
    """Simple database with Turso or SQLite fallback."""
    
    def __init__(self):
        self.use_turso = False
        self._conn = None
        
        # Try Turso first
        if TURSO_URL:
            try:
                import libsql_experimental as libsql
                self._conn = libsql.connect(TURSO_URL, auth_token=TURSO_AUTH_TOKEN)
                self.use_turso = True
                log.info(f"✅ Connected to Turso database")
            except ImportError:
                log.warning("libsql not installed, using SQLite fallback")
            except Exception as e:
                log.warning(f"Turso connection failed: {e}, using SQLite fallback")
        
        # Fallback to SQLite
        if not self.use_turso:
            self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            log.info(f"✅ Using SQLite database: {DB_PATH}")
        
        self._create_tables()
    
    def _create_tables(self):
        """Create required tables."""
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
        """Add features for a Pokémon species."""
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
        log.info(f"✅ Added {len(features)} features for {species}")
    
    def get_all_features(self) -> Dict[str, List[np.ndarray]]:
        """Get all features grouped by species."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT species, feature_vector FROM pokemon_features ORDER BY species, id")
        
        result = {}
        for row in cursor.fetchall():
            species = row[0]
            feature = np.array(json.loads(row[1]))
            result.setdefault(species, []).append(feature)
        return result
    
    def get_all_species(self) -> List[str]:
        """Get all species names."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT species FROM species_info ORDER BY species")
        return [row[0] for row in cursor.fetchall()]
    
    def get_species_count(self, species: str) -> int:
        """Get number of variants for a species."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT count FROM species_info WHERE species = ?", (species,))
        row = cursor.fetchone()
        return row[0] if row else 0
    
    def get_species_features(self, species: str) -> List[np.ndarray]:
        """Get features for a specific species."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT feature_vector FROM pokemon_features WHERE species = ?", (species,))
        return [np.array(json.loads(row[0])) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pokemon_features")
        total_features = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM species_info")
        total_species = cursor.fetchone()[0]
        return {
            "total_features": total_features,
            "total_species": total_species,
            "use_turso": self.use_turso
        }
    
    def close(self):
        if self._conn:
            self._conn.close()


# ============ AI MODEL ============

class PokemonFeatureExtractor(nn.Module):
    """Extract features from Pokémon images."""
    
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
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.to(DEVICE)
        self.eval()
    
    @torch.no_grad()
    def extract(self, img: Image.Image) -> np.ndarray:
        """Extract feature vector from image."""
        img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        img_tensor = self.normalize(img_tensor)
        
        features = self.backbone(img_tensor)
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        
        projected = self.projection(features)
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Extract features from multiple images."""
        batch = []
        for img in images:
            img_tensor = self.transform(img).unsqueeze(0)
            batch.append(img_tensor)
        
        batch_tensor = torch.cat(batch, dim=0).to(DEVICE)
        batch_tensor = self.normalize(batch_tensor)
        
        features = self.backbone(batch_tensor)
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        
        projected = self.projection(features)
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected.cpu().numpy()


class PokemonClassifier(nn.Module):
    """Classifier with feature extractor."""
    
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
        """Forward pass for batch."""
        features = self.feature_extractor.extract_batch(images)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        logits = self.classifier(features_tensor)
        return features_tensor, logits


# ============ DATASET ============

class PokemonDataset(Dataset):
    """Dataset from Hugging Face + local extra Pokémon."""
    
    def __init__(
        self,
        species_to_idx: Dict[str, int],
        max_per_species: int = 10,
        extra_dir: str = "Extra pokemons"
    ):
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
        
        # Try loading from Hugging Face first
        hf_loaded = self._load_huggingface()
        
        # Always load local extras
        self._load_extra_pokemon(extra_dir)
        
        random.shuffle(self.samples)
        self._print_stats()
    
    def _load_huggingface(self):
        """Load images from Hugging Face dataset."""
        log.info(f"📥 Attempting to load Hugging Face dataset: {DATASET_NAME}")
        
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
            
            log.info(f"   Label column: {label_col}, Image column: {image_col}")
            
            species_counts = {}
            count = 0
            for row in ds:
                try:
                    raw_label = row[label_col]
                    if isinstance(raw_label, int):
                        raw_label = features[label_col].int2str(raw_label)
                    
                    species = str(raw_label).strip().lower()
                    if species not in self.species_to_idx:
                        continue
                    
                    if species_counts.get(species, 0) >= self.max_per_species:
                        continue
                    
                    img = row[image_col]
                    if img is None:
                        continue
                    
                    if not isinstance(img, Image.Image):
                        img = Image.open(BytesIO(img))
                    
                    self.samples.append((img, self.species_to_idx[species]))
                    self.species_images.setdefault(species, []).append(img)
                    species_counts[species] = species_counts.get(species, 0) + 1
                    count += 1
                    
                except Exception as e:
                    continue
                
                if count > 1000:  # Limit for speed
                    break
            
            self.species_stats.update(species_counts)
            log.info(f"   ✅ Loaded {len(self.samples)} samples from Hugging Face")
            return True
            
        except Exception as e:
            log.warning(f"   ⚠️ Failed to load Hugging Face dataset: {e}")
            log.warning(f"   Will use only local extra Pokémon images")
            return False
    
    def _load_extra_pokemon(self, extra_dir: str):
        """Load extra Pokémon from local folder."""
        extra_path = Path(extra_dir)
        if not extra_path.exists():
            log.info(f"   No extra Pokémon folder found: {extra_dir}")
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
            
            # Add to species mapping if not exists
            if species not in self.species_to_idx:
                idx = len(self.species_to_idx)
                self.species_to_idx[species] = idx
            
            # Load images
            images = []
            for ext in valid_extensions:
                images.extend(folder.glob(f"*{ext}"))
            
            if not images:
                log.warning(f"   ⚠️ No images found in {folder}")
                continue
            
            # Limit per species
            current_count = species_counts.get(species, 0)
            available = self.max_per_species - current_count
            if available <= 0:
                continue
            
            images = images[:available]
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    
                    self.samples.append((img, self.species_to_idx[species]))
                    self.species_images.setdefault(species, []).append(img)
                    species_counts[species] = species_counts.get(species, 0) + 1
                    
                except Exception as e:
                    log.warning(f"   ⚠️ Failed to load {img_path}: {e}")
        
        # Update stats
        for species, count in species_counts.items():
            self.species_stats[species] = self.species_stats.get(species, 0) + count
        
        log.info(f"   ✅ Loaded {len(self.samples)} samples from extra Pokémon")
    
    def _print_stats(self):
        """Print dataset statistics."""
        log.info("=" * 60)
        log.info("📊 DATASET STATISTICS")
        log.info("=" * 60)
        
        total = len(self.samples)
        species_count = len(self.species_stats)
        
        log.info(f"   Total samples: {total}")
        log.info(f"   Total species: {species_count}")
        
        if self.species_stats:
            counts = list(self.species_stats.values())
            log.info(f"   Avg per species: {sum(counts) / len(counts):.1f}")
            log.info(f"   Min per species: {min(counts)}")
            log.info(f"   Max per species: {max(counts)}")
        
        # Show species with low counts
        low_species = [s for s, c in self.species_stats.items() if c < 3]
        if low_species:
            log.warning(f"\n⚠️ {len(low_species)} species have < 3 images:")
            for s in low_species[:5]:
                log.warning(f"   - {s}: {self.species_stats[s]} images")
            if len(low_species) > 5:
                log.warning(f"   ... and {len(low_species) - 5} more")
        
        # Show extra Pokémon loaded
        extra_path = Path("Extra pokemons")
        if extra_path.exists():
            extra_species = [f.name for f in extra_path.iterdir() if f.is_dir()]
            if extra_species:
                log.info(f"\n📁 Extra Pokémon loaded: {len(extra_species)} species")
                for s in extra_species[:5]:
                    count = self.species_stats.get(s.lower().replace("_", " ").strip(), 0)
                    log.info(f"   - {s}: {count} images")
                if len(extra_species) > 5:
                    log.info(f"   ... and {len(extra_species) - 5} more")
        
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
    """Main training function."""
    log.info("")
    log.info("🚀 Pokémon AI Trainer")
    log.info("=" * 60)
    
    # Extract any archive files
    log.info("📦 Checking for archive files (ZIP, RAR, 7z, TAR)...")
    extract_archive_files()
    
    # Initialize database
    log.info("\n📂 Connecting to database...")
    db = Database()
    stats = db.get_stats()
    log.info(f"   Existing species: {stats['total_species']}")
    log.info(f"   Existing features: {stats['total_features']}")
    
    # Check for extra Pokémon
    extra_path = Path("Extra pokemons")
    extra_species = []
    if extra_path.exists():
        extra_species = [f.name.replace("_", " ").strip().lower() 
                         for f in extra_path.iterdir() if f.is_dir()]
        log.info(f"📁 Found {len(extra_species)} extra Pokémon species")
        
        # Show what was found
        for s in extra_species[:5]:
            folder = extra_path / s.replace(" ", "_")
            count = len(list(folder.glob("*"))) if folder.exists() else 0
            log.info(f"      - {s}: {count} images")
        if len(extra_species) > 5:
            log.info(f"      ... and {len(extra_species) - 5} more")
    
    # If no species found, exit
    if not extra_species:
        log.error("❌ No extra Pokémon found! Please upload images.")
        log.error("   Expected: Extra pokemons/pikachu/1.png")
        return
    
    # Use only extra species (skip Hugging Face if it failed)
    all_species = sorted(extra_species)
    log.info(f"   Total species: {len(all_species)}")
    
    # Create mapping
    species_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_species = {i: s for s, i in species_to_idx.items()}
    
    # Create dataset
    log.info("\n📂 Creating dataset...")
    dataset = PokemonDataset(
        species_to_idx=species_to_idx,
        max_per_species=MAX_IMAGES_PER_SPECIES,
        extra_dir="Extra pokemons"
    )
    
    if len(dataset) == 0:
        log.error("❌ No samples loaded! Exiting.")
        return
    
    # Split dataset
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    log.info(f"   Train samples: {train_size}")
    log.info(f"   Val samples: {val_size}")
    
    # Initialize model
    log.info("\n🧠 Initializing model...")
    model = PokemonClassifier(num_species=len(all_species))
    model.to(DEVICE)
    model.feature_extractor.eval()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    log.info(f"\n🎯 Training for {EPOCHS} epochs...")
    log.info(f"   Batch size: {BATCH_SIZE}")
    log.info(f"   Learning rate: {LEARNING_RATE}")
    log.info("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_imgs, batch_labels in progress_bar:
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Convert tensors back to PIL for feature extraction
            pil_imgs = []
            for i in range(batch_imgs.size(0)):
                img = batch_imgs[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                img = transforms.ToPILImage()(img)
                pil_imgs.append(img)
            
            features, logits = model.forward_batch(pil_imgs)
            features = features.to(DEVICE)
            logits = logits.to(DEVICE)
            
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            
            progress_bar.set_postfix({
                "loss": f"{train_loss/train_total:.3f}",
                "acc": f"{100*train_correct/train_total:.1f}%"
            })
        
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
                    img = batch_imgs[i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1)
                    img = transforms.ToPILImage()(img)
                    pil_imgs.append(img)
                
                _, logits = model.forward_batch(pil_imgs)
                _, predicted = torch.max(logits, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        train_acc = 100 * train_correct / train_total
        
        log.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        
        scheduler.step()
        
        # Save best model
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
    
    # Collect all images by species from dataset
    species_images = {}
    for img, label_idx in dataset.samples:
        species = idx_to_species[label_idx]
        species_images.setdefault(species, []).append(img)
    
    log.info(f"   Extracting features for {len(species_images)} species...")
    
    for species, images in tqdm(species_images.items(), desc="   Storing"):
        try:
            features = []
            for img in images:
                if isinstance(img, Image.Image):
                    feature = model.feature_extractor.extract(img)
                    features.append(feature)
            
            if features:
                db.add_pokemon_features(species, features)
        except Exception as e:
            log.error(f"   Failed to store {species}: {e}")
    
    # Save species info
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
    
    # Final stats
    final_stats = db.get_stats()
    log.info(f"\n📊 Database Stats:")
    log.info(f"   Total species: {final_stats['total_species']}")
    log.info(f"   Total features: {final_stats['total_features']}")
    log.info(f"   Using Turso: {final_stats['use_turso']}")
    
    log.info("\n" + "=" * 60)
    log.info("✅ All done! Model is ready to use.")
    log.info("=" * 60)
    
    # Close database
    db.close()


if __name__ == "__main__":
    train()
