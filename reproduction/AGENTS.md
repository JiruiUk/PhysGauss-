# Training pause and resume policy

The user explicitly requested saving training state to disk and releasing GPU memory while paused. Honor this for future pause requests. Do not use Windows process suspension to retain GPU allocations unless the user explicitly asks for that behavior again.

The trainer now saves full checkpoints, including both optimizers and random states, at `runs/pidg_official/<scene>/checkpoints/resume.pt`. Inference PLY/material/deformation snapshots are separate and cannot restore an optimizer.

To stop an active run, use `scripts/save_and_stop.ps1 -Scene <scene>`. It requests an end-of-iteration checkpoint. Trainer exit code 75 tells the controller to mark the scene `saved_stopped` and stop the remaining queue without rendering or starting the next scene. Verify the saved manifest/checksum and process exit before claiming GPU release. If saving is still pending, preserve the live process rather than terminating it.

To continue the current remaining queue on a later user request:

```powershell
& 'F:\A_Edinburgh\PhysGauss\reproduction\scripts\run_reproduction.ps1' -Scenes motion_kuro,rubber_duck -Iterations 50000 -Profile official -Resume
```

Do not restart from zero when a verified full checkpoint exists. Keep completed scenes and existing logs. `-Resume` appends to the training log and restores the saved step, Gaussian/deformation/material parameters, optimizer states, camera sampling order and random states. No training should be started as part of a save/release request.
