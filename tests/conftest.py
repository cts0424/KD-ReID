import os
import sys
from pathlib import Path

# Synthetic datasets never match the published split sizes; skip that check in every test
# (also inherited by tools launched as subprocesses).
os.environ["KDREID_SKIP_DATASET_CHECK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
