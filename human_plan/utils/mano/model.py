import torch
import numpy as np
import smplx
from pathlib import Path
from smplx.lbs import blend_shapes, vertices2joints

MANO_MODEL_DIR = Path(__file__).resolve().parents[3] / "mano_v1_2" / "models"

# Older MANO pickles pull in chumpy, which still expects NumPy's deprecated
# scalar aliases. Recreate them for modern NumPy versions used in eval envs.
NUMPY_LEGACY_ALIASES = {
  "bool": np.bool_,
  "int": int,
  "float": float,
  "complex": complex,
  "object": object,
  "unicode": str,
  "str": str,
}
for alias_name, alias_value in NUMPY_LEGACY_ALIASES.items():
  if not hasattr(np, alias_name):
    setattr(np, alias_name, alias_value)

mano_left = smplx.create(
  str(MANO_MODEL_DIR / "MANO_LEFT.pkl"),
  "mano",
  use_pca=True,
  is_rhand=False,
  num_pca_comps=15,
)
mano_left.to("cpu")

mano_right = smplx.create(
  str(MANO_MODEL_DIR / "MANO_RIGHT.pkl"),
  "mano",
  use_pca=True,
  is_rhand=True,
  num_pca_comps=15,
)
mano_right.to("cpu")
