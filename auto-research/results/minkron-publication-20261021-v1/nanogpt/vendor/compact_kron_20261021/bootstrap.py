"""Use the private CPU runtime when present; ordinary installations also work."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
runtime=Path(os.environ.get('KRON_RUNTIME_PATH',str(ROOT.parent/'kron-integrated-runtime')))
if runtime.exists():sys.path.insert(0,str(runtime))
sys.path.insert(0,str(ROOT/'upstream'))
