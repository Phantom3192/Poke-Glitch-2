"""
train_model.py - Complete AI Pokémon Trainer for Railway
Fetches from Hugging Face + your extra Pokémon, trains model, stores in database
"""

import os
import sys
import json
import logging
import sqlite3
import random
import time
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
from datasets import load_dataset

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

# Device
DEVICE = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("trainer")

# ============ DATABASE LAYER ============

class Database:
    """Simple database with Turso or SQLite fallback."""
    
    def __init__(self):
        self.use_turso = False
        self._conn = None
        
        # Try Turso first
        if TURSO_URL:
            try:
                # Try importing libsql
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
        self.species_images = {}  # Store images for feature extraction
        
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
        
        # Load extra Pokémon
        self._load_extra_pokemon(extra_dir)
        
        random.shuffle(self.samples)
        self._print_stats()
    
    def _load_huggingface(self):
        """Load images from Hugging Face dataset."""
        log.info(f"📥 Loading Hugging Face dataset: {DATASET_NAME}")
        
        try:
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
            for row in ds:
                try:
                    # Get label
                    raw_label = row[label_col]
                    if isinstance(raw_label, int):
                        raw_label = features[label_col].int2str(raw_label)
                    
                    species = str(raw_label).strip().lower()
                    if species not in self.species_to_idx:
                        continue
                    
                    if species_counts.get(species, 0) >= self.max_per_species:
                        continue
                    
                    # Get image
                    img = row[image_col]
                    if img is None:
                        continue
                    
                    if not isinstance(img, Image.Image):
                        img = Image.open(BytesIO(img))
                    
                    self.samples.append((img, self.species_to_idx[species]))
                    self.species_images.setdefault(species, []).append(img)
                    species_counts[species] = species_counts.get(species, 0) + 1
                    
                except Exception as e:
                    continue
            
            self.species_stats.update(species_counts)
            log.info(f"   ✅ Loaded {len(self.samples)} samples from Hugging Face")
            
        except Exception as e:
            log.error(f"   ❌ Failed to load Hugging Face dataset: {e}")
    
    def _load_extra_pokemon(self, extra_dir: str):
        """Load extra Pokémon from local folder."""
        extra_path = Path(extra_dir)
        if not extra_path.exists():
            log.info(f"   No extra Pokémon folder found: {extra_dir}")
            return
        
        log.info(f"📁 Loading extra Pokémon from: {extra_dir}")
        
        valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        species_counts = {}
        
        for folder in extra_path.iterdir():
            if not folder.is_dir():
                continue
            
            species = folder.name.replace("_", " ").strip().lower()
            if species not in self.species_to_idx:
                log.warning(f"   ⚠️ Species '{species}' not in species list, adding...")
                # Add to species mapping
                idx = len(self.species_to_idx)
                self.species_to_idx[species] = idx
                # Update dataset samples label mapping
                # We'll handle this after loading all samples
            
            # Load images
            images = []
            for ext in valid_extensions:
                images.extend(folder.glob(f"*{ext}"))
            
            if not images:
                log.warning(f"   ⚠️ No images found in {folder}")
                continue
            
            # Limit per species (don't exceed max)
            current_count = species_counts.get(species, 0)
            available = self.max_per_species - current_count
            if available <= 0:
                continue
            
            images = images[:available]
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    # Get or create index for this species
                    if species not in self.species_to_idx:
                        idx = len(self.species_to_idx)
                        self.species_to_idx[species] = idx
                    
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
    
    # Initialize database
    log.info("📂 Connecting to database...")
    db = Database()
    stats = db.get_stats()
    log.info(f"   Existing species: {stats['total_species']}")
    log.info(f"   Existing features: {stats['total_features']}")
    
    # Get species from Hugging Face
    log.info("\n🔍 Discovering species...")
    hf_species = set()
    try:
        ds = load_dataset(DATASET_NAME, split="train", streaming=True)
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
                if count > 5000:  # Enough to get most species
                    break
        log.info(f"   Found {len(hf_species)} species in Hugging Face")
    except Exception as e:
        log.warning(f"   Could not get species list: {e}")
    
    # Check for extra Pokémon
    extra_path = Path("Extra pokemons")
    extra_species = []
    if extra_path.exists():
        extra_species = [f.name.replace("_", " ").strip().lower() 
                         for f in extra_path.iterdir() if f.is_dir()]
        log.info(f"   Found {len(extra_species)} extra Pokémon species")
    
    # Combine species
    all_species = sorted(hf_species | set(extra_species))
    log.info(f"   Total species: {len(all_species)}")
    
    if not all_species:
        log.error("❌ No species found! Exiting.")
        return
    
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


if __name__ == "__main__":
    train()
