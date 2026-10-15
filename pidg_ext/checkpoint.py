"""Extend the original complete checkpoint without changing flags-off snapshots."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from reproduction.scripts import training_checkpoint as baseline


def replace_with_retry(source,destination,attempts=100,delay=.05):
    """Windows readers can briefly deny delete sharing during atomic publication."""
    for attempt in range(attempts):
        try:
            return Path(source).replace(destination)
        except PermissionError:
            if attempt==attempts-1:raise
            time.sleep(delay)


def save_training_state(local,destination):
    manifest=baseline.save_training_state(local,destination)
    extension=local.get('extension')
    if extension is None or not extension.enabled:return manifest
    destination=Path(destination)
    state=baseline.load_state(destination)
    state['constitutive_extension']=extension.checkpoint_state()
    temporary=destination.with_suffix('.extension.tmp')
    with temporary.open('wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    replace_with_retry(temporary,destination)
    with destination.open('rb') as stream:
        manifest['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    manifest.update(size_bytes=destination.stat().st_size,extension_saved=True)
    sidecar=destination.with_suffix('.json');temporary=sidecar.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest,indent=2),encoding='utf-8');replace_with_retry(temporary,sidecar)
    print('EXTENSION_CHECKPOINT_SAVED '+json.dumps(manifest),flush=True)
    return manifest


def restore_training_state(path,gaussians,deform,opt,scene,extension=None):
    state=baseline.load_state(path)
    if extension is not None and extension.enabled and 'constitutive_extension' not in state:
        raise ValueError('Cannot resume enabled extension without its complete saved state')
    restored=baseline.restore_training_state(path,gaussians,deform,opt,scene)
    if extension is not None:
        extension.restore_checkpoint(state.get('constitutive_extension'),gaussians.get_xyz.device)
    return restored
