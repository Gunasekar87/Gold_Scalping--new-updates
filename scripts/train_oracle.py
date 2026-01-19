import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_core.nexus_trainer import NexusTrainer

if __name__ == "__main__":
    print("Initializing Nexus Trainer...")
    trainer = NexusTrainer()
    print("Training Model...")
    loss = trainer.train(epochs=5)
    print(f"Training Complete. Final Loss: {loss:.4f}")
