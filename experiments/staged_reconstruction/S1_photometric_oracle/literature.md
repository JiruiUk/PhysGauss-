# Measurement model, after D4's fixed proposal rejection

- [Bleyer, Rhemann, Rother, PatchMatch Stereo, BMVC2011](https://bmva-archive.org.uk/bmvc/2011/proceedings/paper14/index.html),
  [author-hosted full paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/PatchMatchStereo_BMVC2011_6MB.pdf).
  Read introduction and method sections2.1–2.2. Matching a fixed support window
  can be biased by slanted surfaces and mixed-depth boundaries; the paper fits
  planes and warps support accordingly. Our normalized5px feature sampler is
  different. This motivates a measurement audit; it does not prove either bias
  is our dominant error. No plane search or paper hyperparameter is copied.
- [Xu and Tao, Multi-Scale Geometric Consistency Guided Multi-View Stereo,
  CVPR2019 official record](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Multi-Scale_Geometric_Consistency_Guided_Multi-View_Stereo_CVPR_2019_paper.html).
  Verified official abstract: adaptive view selection and multi-scale geometric
  consistency address unreliable depth in weak texture. Official full PDF403;
  no full-method reading claim. Our held-source checks are not independent GT,
  and unknown/occluded support must not become dynamic/background labels.

D4's one changed point also improved held RGB costs while increasing true
depth error. Audit original cost at privileged true depths and stratify source
visibility/reference support before choosing a new mechanism. This is distinct
from copying a stereo method into PIDG or declaring that paper reproduced.
