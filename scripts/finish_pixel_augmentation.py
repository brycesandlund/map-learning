"""Wait for the bounded tuning sweep, apply its frozen gate, run confirmation once."""
import json,subprocess,sys,time
from pathlib import Path
root=Path('runs/pixel-attention-augmentation-20261111')
while not (root/'complete.json').exists():time.sleep(10)
status=json.loads((root/'complete.json').read_text());assert status['failed']==0,status
subprocess.run([sys.executable,'scripts/select_pixel_augmentation.py'],check=True)
selection=json.loads(Path('configs/pixel-study-20261111/augmentation-selection.json').read_text())
if selection['selected']:
    subprocess.run([sys.executable,'scripts/run_pixel_study.py','--arms','configs/pixel-study-20261111/augmentation-confirm.json','--out','runs/pixel-augmentation-confirm-20261111','--seeds','130','131','132','--workers','3'],check=True)
