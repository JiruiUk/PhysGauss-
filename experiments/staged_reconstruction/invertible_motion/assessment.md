# I7: invertibility alone does not restore complete geometry

Actual16,000-step warmup completed from the original initial state, with
original12k800+4k1600 schedule and sampler, plus33,420 new shared position
parameters. Original parameter tensors/Adam groups were retained; the
original warp head was inactive, and covariance heads remained active.
No new depth, tracking, strain or constitutive objective was used.

Validation30: full PSNR27.453076, dynamic PSNR25.453224, dynamic alpha0.980829,
static PSNR34.317455. Against frozen static-only35.643816, static drop1.326361dB
fails the original1dB warmup gate. No formal physics updates were admitted.
The independent warmup audit verifies original initialization, full16k sampler,
CPU/CUDA/Python/NumPy RNG, finite Adam states, inactive original warp head and
unchanged static parameters. Its exact checkpoint hash is in that receipt.

Actual1600x900 review: fixed188/t43 has incomplete surfaces, detached yellow
particles and elongated bridges; t146 retains severe stretching and missing
regions, including a long dark streak; fixed86/t86 is smoother but still has
detached support. The visible failures reject completeness independently of
the static gate. No22k completion, full1200-frame scan or test-RGB evaluation.
Preserve the failed original admission; actual image review is separate.

An invertible map has positive determinant but may have large shear and
strongly unequal singular values. It also does not constrain Gaussian
covariance to that map or certify a complete canonical shape. Before another
training objective, measure local strain of this actual failed checkpoint;
only use a constitutive/strain hypothesis if that diagnostic supports it.
