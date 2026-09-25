"""The frozen 48-candidate PEMix search space."""
import json
from pathlib import Path

def pemix_candidates():
    payload=json.loads((Path(__file__).parent/'configs/extended_search.json').read_text())
    return payload
