### Player Tracking Report

Note: This submission is self-contained and documented for reproducibility. No external debugging is expected, but you must update the file paths (`VIDEO_PATH`, `MODEL_PATH`, and `OUTPUT_PATH`) in the code (`player_identify.py`) to match your local environment before running.

This report walks through how I built and tested a system to track and re‑identify players in a sports video, what worked (and what didn't), the hurdles I hit, and how I would move forward with more time or resources.

1. What I Did

I loaded my custom YOLOv8 weights (best.pt), then set up a simple pipeline that:

Spotted players in each frame with YOLOv11 (Ultralytics).

Grabbed their 'look' using a pre-trained Vision Transformer (ViT) for feature embeddings.

Matched new detections to existing tracks by blending three cues: how much boxes overlap (IoU), how similar they look (embedding distance), and how far apart their centers are.

Kept tracks alive with a Kalman filter and the Hungarian algorithm for assignment.

I tuned parameters like confidence threshold (0.4), minimum box size, and fusion weights (30% IoU, 50% appearance, 20% distance) until it felt pretty stable.

2. What I Tried & What Happened

Only IoU: super fast but lost IDs when players crossed or got close.

Only appearance: better at some overlaps, but swapped IDs when players looked alike.

Equal-weight fusion: a bit better, but still wobbly during quick moves.

Weighted fusion (0.3/0.5/0.2): my sweet spot: fewer ID swaps, decent speed.

Resizing frames to 640×360: doubled speed but tracking got slightly blurrier.

3. Hurdles & Oddities

Random flying boxes: sometimes a box would flash on screen (originating from a player) and shoot off in the next frame. I think YOLO was jittering under motion blur.

ID swaps in crowds: at the goal post, when players cluster, a few IDs flipped unexpectedly.

Compute limits: without more GPU time, I couldn’t train a custom re‑ID model or test on really long videos.

4. What’s Left & Next Steps

If I had more time or resources, I would:

Smooth detections with a temporal filter to kill off quick flickers.

Train a re‑ID network (e.g. ResNet with triplet loss) on sports images for stronger appearance matching.

Handle occlusions by predicting motion paths and filling in missing detections.

Group players in clusters with graph-based tracking so IDs stay consistent when they bunch up.

Optimize for speed by quantizing the model and fully using GPU.

In a nutshell, the core tracking works well, but there are still some limitations with brief false detections and crowded scenes. With a bit more tuning and data, I am confident it will be rock solid.
