# Browser video playback correction

The original review exporter used OpenCV `mp4v` (MPEG-4 Part 2) in MP4.
All frames decoded in the independent audit, but that did not establish HTML5
browser codec compatibility. The existing Chrome gallery could therefore show
unplayable media despite valid files and paths.

`prepare_browser_videos.py` creates separate `.browser.mp4` display copies with
H.264 Main level3.1, yuv420p and the MP4 metadata before media data (`faststart`).
Every copy independently decodes200 frames at20fps and800x450. The conversion
receipt is `browser_video_conversion.json`. Original exports and their hashes
in `completion_integrity.json` are preserved. Lossy display transcoding is not
used to recalculate reconstruction metrics or select model checkpoints.

The gallery generator now uses these copies with explicit `video/mp4` sources
and direct video links. Its fifteen local media links are checked separately.
Browser UI inspection of this local `file:` page was blocked by the browser
tool's protocol policy, so no claim of automated browser playback is made.
Full decoding, H.264 profile/pixel-format/container metadata and page links are
verified without changing browser security settings or bypassing that block.

The first conversion validation expected OpenCV's FOURCC to report `avc1`;
this build reports `h264` instead. The validation receipt and first output are
retained; the corrected check accepts the decoder identifier and independently
checks the actual avc1/H.264 Main/yuv420p stream metadata in FFmpeg's log.
