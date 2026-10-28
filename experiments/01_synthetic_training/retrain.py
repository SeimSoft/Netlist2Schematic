"""Retrain the policy on the whole training database (same configuration as the UI)."""
import time

from netlist2schematic import paths, trainer
from netlist2schematic.training_db import TrainingDB

start = time.time()
history = trainer.train(TrainingDB(paths.db_path()), paths.model_path(), trainer.TrainConfig(),
                        progress=lambda stage, done, total, info: print(f"{stage} {done}/{total} {info}", flush=True) if done % 20 == 0 or done == total else None)
print(f"trained in {time.time() - start:.0f}s, loss {history['bc_loss'][0]:.3f} -> {history['bc_loss'][-1]:.3f}", flush=True)
