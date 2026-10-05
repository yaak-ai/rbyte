# nero robot-native ingestion (patch family, contract v3)

`rbyte.samples.nero.NeroRobotReader` + `rbyte.samples.nero.NeroRobotWindowGrouper` turn the
ROBOT recordings (nutron-cli `record.py`: `data.mcap` + `base/side_left/side_right.mp4`
+ `outcome.json`) into training samples for rmind's robot-native `NeroPatchPolicy`.
The glove recordings (`NeroArmsReader`) are a different source and are
unchanged.

## Clock and alignment: a port of nutron-cli `convert.py`

rbyte cannot import nutron-cli, so `src/rbyte/samples/nero/robot.py` ports the rules
that build the ACT dataset, and `tests/test_nero_robot.py` pins the port
bit-for-bit against `convert.align` on the pulled episodes (when present):

* `t` = base camera `publish_time` (host clock, mid-exposure), int64 ns;
* arm measured/command and hand command: zero-order hold at `t`, age <= 100 ms
  and `t <= last sample`;
* `hand_prev` = hand command ZOH at `t - 1/30 s` (age <= 100 ms);
* side cameras: nearest frame within 50 ms;
* valid spans shorter than 100 frames are dropped;
* a recording whose stored camera names are swapped (camera serials) is
  refused rather than silently fed the wrong view.

## Row = one base frame (30 Hz)

| column | shape | meaning |
|---|---|---|
| `state` | `(2, 13)` | measured q (7) + `hand_prev / 1000` (6); invalid side all 0 |
| `action.chunk` | `(100, 2, 13)` | `chunk[k]` = command q + hand command / 1000 at `t + k/30 s`; `k = 0` is convert's `action` |
| `action.is_pad` | `(100,)` | hold-padded steps past the end of the row's valid run |
| `side_valid` | `(2,)` | `episode.active_hand` |
| `hand.current/pos_err/pos/tip/age/motor_ok/tip_ok` | | the vendored `hand_features.build_tokens` blocks (newest sample) |
| `grid_index` | | gap-aware 30 Hz counter |
| `frame_index.{camera}` | | mp4 ordinals |
| `camera_cond` | `(3, 13)` | zeros + `placeholder` (no calibration on the robot rig yet) |

* **Padding.** Chunks are bounded by the row's valid run (episode end, a stale
  gap, a reset span): the last real command is held and `action.is_pad` set;
  rows with more than `chunk_size // 2` padded steps are dropped. Without it a
  100-step chunk would drop the last 3.3 s (the release) of every run.
* **Hand.** Never NULL: a refused/stale hand reading is zeros with
  `hand.motor_ok = False`, and the model substitutes `no_hand`. Only reset spans
  (recorder record UNION self-detected) remove rows.

## Windows = one sample (10 Hz)

`NeroRobotWindowGrouper(clip_frames=T, frame_stride=3, episode_stride, episode_offset)`
emits a window starting at grid index `g` only if EVERY grid index
`g .. g + 3(T-1)` exists, then gathers `g, g+3, ...`. A dropped camera frame, a
stale-state gap or a reset span therefore can never be bridged with a 66/133 ms
step that the KV/RoPE model would read as 100 ms (`gather_every` on rows would).
Choose `episode_stride` coprime to 3 so the starts cycle through all 30 Hz phases.

## hand_features: a pinned verbatim copy

`src/rbyte/samples/nero/_vendor/hand_features.py` is nutron-cli's
`runtime/jetson/hand_features.py`, byte for byte, pinned by
`HAND_FEATURES_SHA256` (+ both version constants) in `_vendor/__init__.py`.
A path import was rejected: rbyte runs where no nutron-cli checkout exists.
Never edit the copy. Resync after any nutron-cli edit (a comment included):

```sh
python <nutron-cli>/runtime/jetson/hand_features_sync.py --vendor src/rbyte/samples/nero/_vendor
# paste the three printed constants into src/rbyte/samples/nero/_vendor/__init__.py
```

`tests/test_nero_robot.py` always checks the hash and versions, and compares
against the nutron-cli file when `NUTRON_CLI_ROOT` (default
`/home/max/Code/nutron-cli-patch-policy`) exists. A semantic change to the token
bumps `HAND_TOKEN_SPEC_VERSION` and needs a rebuilt cache, a retrain and a
re-export.

## Images

The robot dataset config (rmind `config/_templates/dataset/nero/robot.lib.yml`)
decodes NATIVE frames and wraps the source in `rbyte.streams.transformed.TransformedSource`
with rmind's `rmind.data.nero_image.NeroImagePreprocess` -- the same function
serving calls -- instead of a decoder-side (swscale) resize.
