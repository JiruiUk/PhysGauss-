"""The reconstruction-facing input boundary: RGB, camera and time only."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def load(directory):
    directory = Path(directory).resolve()
    meta = json.loads((directory/'cameras.json').read_text(encoding='utf-8'))
    assert set(meta) == {'width','height','K','frames'}
    images,views,times = [],[],[]
    for row in meta['frames']:
        assert set(row) == {'image','world_to_camera','time'}
        path = (directory/row['image']).resolve()
        assert path.is_relative_to(directory/'rgb') and path.suffix == '.png'
        with Image.open(path) as image:
            assert image.mode == 'RGB' and image.size == (meta['width'],meta['height'])
            images.append(np.asarray(image,dtype=np.float32)/255)
        views.append(row['world_to_camera']); times.append(row['time'])
    return {'rgb':np.asarray(images),'views':np.asarray(views),'times':np.asarray(times),
            'K':np.asarray(meta['K'])}
