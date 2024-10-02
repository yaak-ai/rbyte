<p align="center">
 <a href="https://www.yaak.ai/open-source/dev-tools">
  <img alt="banner" src="https://github.com/user-attachments/assets/707ab3ae-73d5-459f-82c5-888323673adb">
 </a>
</p>

<p align="center">
 <img src="https://github.com/yaak-ai/rbyte/actions/workflows/ci.yaml/badge.svg">
 <img src="https://img.shields.io/github/license/yaak-ai/rbyte.svg?color=green"></a>
</p>


`rbyte` provides a [PyTorch](https://pytorch.org) [`Dataset`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) with [`tensorclass`](https://pytorch.org/tensordict/main/reference/tensorclass.html) samples built from multimodal data

## Installation

```bash
uv add https://github.com/yaak-ai/rbyte/releases/latest/download/rbyte-X.Y.Z-py3-none-any.whl [--extra mcap] [--extra jpeg] [--extra visualize]
```

## Examples

See [examples/config_templates](examples/config_templates) ([`ytt`](https://carvel.dev/ytt/) templates) and [justfile](justfile) for usage examples.

<details>
<summary><a href=https://nuscenes.org> nuScenes </a> x <a href=https://mcap.dev> mcap </a></summary>

Setup a new project with [`uv`](https://docs.astral.sh/uv/):
```shell
uv init nuscenes_mcap
cd nuscenes_mcap

uv add hydra-core omegaconf
uv add https://github.com/yaak-ai/rbyte/releases/latest/download/rbyte-0.3.0-py3-none-any.whl --extra mcap --extra jpeg --extra visualize

mkdir data
```

Follow the guide at [foxglove/nuscenes2mcap](https://github.com/foxglove/nuscenes2mcap) and move the resulting `.mcap` files under `data/`. In this example we're using a subset of topics from `NuScenes-v1.0-mini-scene-0103.mcap`:
```shell
mcap info data/NuScenes-v1.0-mini-scene-0103.mcap
library:   nuscenes2mcap
profile:
messages:  34764
duration:  19.443428s
start:     2018-08-01T21:26:43.504799+02:00 (1533151603.504799000)
end:       2018-08-01T21:27:02.948227+02:00 (1533151622.948227000)
compression:
        lz4: [629/629 chunks] [753.36 MiB/481.51 MiB (36.09%)] [24.76 MiB/sec]
channels:
        (1)  /imu                                     1933 msgs (99.42 Hz)     : IMU [jsonschema]
        (2)  /odom                                     968 msgs (49.79 Hz)     : Pose [jsonschema]
        (3)  /map                                        1 msgs                : foxglove.Grid [protobuf]
        (4)  /semantic_map                               1 msgs                : foxglove.SceneUpdate [protobuf]
        (5)  /tf                                      3103 msgs (159.59 Hz)    : foxglove.FrameTransform [protobuf]
        (6)  /drivable_area                             40 msgs (2.06 Hz)      : foxglove.Grid [protobuf]
        (7)  /RADAR_FRONT                              266 msgs (13.68 Hz)     : foxglove.PointCloud [protobuf]
        (8)  /RADAR_FRONT_LEFT                         258 msgs (13.27 Hz)     : foxglove.PointCloud [protobuf]
        (9)  /RADAR_FRONT_RIGHT                        259 msgs (13.32 Hz)     : foxglove.PointCloud [protobuf]
        (10) /RADAR_BACK_LEFT                          252 msgs (12.96 Hz)     : foxglove.PointCloud [protobuf]
        (11) /RADAR_BACK_RIGHT                         255 msgs (13.11 Hz)     : foxglove.PointCloud [protobuf]
        (12) /LIDAR_TOP                                389 msgs (20.01 Hz)     : foxglove.PointCloud [protobuf]
        (13) /CAM_FRONT/image_rect_compressed          229 msgs (11.78 Hz)     : foxglove.CompressedImage [protobuf]
        (14) /CAM_FRONT/camera_info                    229 msgs (11.78 Hz)     : foxglove.CameraCalibration [protobuf]
        (15) /CAM_FRONT/lidar                          229 msgs (11.78 Hz)     : foxglove.ImageAnnotations [protobuf]
        (16) /CAM_FRONT/annotations                     40 msgs (2.06 Hz)      : foxglove.ImageAnnotations [protobuf]
        (17) /CAM_FRONT_RIGHT/image_rect_compressed    233 msgs (11.98 Hz)     : foxglove.CompressedImage [protobuf]
        (18) /CAM_FRONT_RIGHT/camera_info              233 msgs (11.98 Hz)     : foxglove.CameraCalibration [protobuf]
        (19) /CAM_FRONT_RIGHT/lidar                    233 msgs (11.98 Hz)     : foxglove.ImageAnnotations [protobuf]
        (20) /CAM_FRONT_RIGHT/annotations               40 msgs (2.06 Hz)      : foxglove.ImageAnnotations [protobuf]
        (21) /CAM_BACK_RIGHT/image_rect_compressed     234 msgs (12.03 Hz)     : foxglove.CompressedImage [protobuf]
        (22) /CAM_BACK_RIGHT/camera_info               234 msgs (12.03 Hz)     : foxglove.CameraCalibration [protobuf]
        (23) /CAM_BACK_RIGHT/lidar                     234 msgs (12.03 Hz)     : foxglove.ImageAnnotations [protobuf]
        (24) /CAM_BACK_RIGHT/annotations                40 msgs (2.06 Hz)      : foxglove.ImageAnnotations [protobuf]
        (25) /CAM_BACK/image_rect_compressed           229 msgs (11.78 Hz)     : foxglove.CompressedImage [protobuf]
        (26) /CAM_BACK/camera_info                     229 msgs (11.78 Hz)     : foxglove.CameraCalibration [protobuf]
        (27) /CAM_BACK/lidar                           229 msgs (11.78 Hz)     : foxglove.ImageAnnotations [protobuf]
        (28) /CAM_BACK/annotations                      40 msgs (2.06 Hz)      : foxglove.ImageAnnotations [protobuf]
        (29) /CAM_BACK_LEFT/image_rect_compressed      228 msgs (11.73 Hz)     : foxglove.CompressedImage [protobuf]
        (30) /CAM_BACK_LEFT/camera_info                228 msgs (11.73 Hz)     : foxglove.CameraCalibration [protobuf]
        (31) /CAM_BACK_LEFT/lidar                      228 msgs (11.73 Hz)     : foxglove.ImageAnnotations [protobuf]
        (32) /CAM_BACK_LEFT/annotations                 40 msgs (2.06 Hz)      : foxglove.ImageAnnotations [protobuf]
        (33) /CAM_FRONT_LEFT/image_rect_compressed     231 msgs (11.88 Hz)     : foxglove.CompressedImage [protobuf]
        (34) /CAM_FRONT_LEFT/camera_info               231 msgs (11.88 Hz)     : foxglove.CameraCalibration [protobuf]
        (35) /CAM_FRONT_LEFT/lidar                     231 msgs (11.88 Hz)     : foxglove.ImageAnnotations [protobuf]
        (36) /CAM_FRONT_LEFT/annotations                40 msgs (2.06 Hz)      : foxglove.ImageAnnotations [protobuf]
        (37) /pose                                      40 msgs (2.06 Hz)      : foxglove.PoseInFrame [protobuf]
        (38) /gps                                       40 msgs (2.06 Hz)      : foxglove.LocationFix [protobuf]
        (39) /markers/annotations                       40 msgs (2.06 Hz)      : foxglove.SceneUpdate [protobuf]
        (40) /markers/car                               40 msgs (2.06 Hz)      : foxglove.SceneUpdate [protobuf]
        (41) /diagnostics                            22487 msgs (1156.53 Hz)   : diagnostic_msgs/DiagnosticArray [ros1msg]
attachments: 0
metadata: 1
```

Create a `config.yaml` to extract frames from three cameras + velocity, aligning everything to the first camera's timestamp:
```yaml
---
dataloader:
  _target_: torch.utils.data.DataLoader
  dataset: ${dataset}
  batch_size: 1
  collate_fn:
    _target_: rbyte.utils.dataloader.collate_identity
    _partial_: true

dataset:
  _target_: rbyte.Dataset
  _convert_: all
  _recursive_: false
  inputs:
    NuScenes-v1.0-mini-scene-0103:
      frame:
        /CAM_FRONT/image_rect_compressed:
          index_column: /CAM_FRONT/image_rect_compressed/idx
          reader:
            _target_: rbyte.io.frame.mcap.McapFrameReader
            path: data/NuScenes-v1.0-mini-scene-0103.mcap
            topic: /CAM_FRONT/image_rect_compressed
            decoder_factory: mcap_protobuf.decoder.DecoderFactory
            frame_decoder: ${frame_decoder}

        /CAM_FRONT_LEFT/image_rect_compressed:
          index_column: /CAM_FRONT_LEFT/image_rect_compressed/idx
          reader:
            _target_: rbyte.io.frame.mcap.McapFrameReader
            path: data/NuScenes-v1.0-mini-scene-0103.mcap
            topic: /CAM_FRONT_LEFT/image_rect_compressed
            decoder_factory: mcap_protobuf.decoder.DecoderFactory
            frame_decoder: ${frame_decoder}

        /CAM_FRONT_RIGHT/image_rect_compressed:
          index_column: /CAM_FRONT_RIGHT/image_rect_compressed/idx
          reader:
            _target_: rbyte.io.frame.mcap.McapFrameReader
            path: data/NuScenes-v1.0-mini-scene-0103.mcap
            topic: /CAM_FRONT_RIGHT/image_rect_compressed
            decoder_factory: mcap_protobuf.decoder.DecoderFactory
            frame_decoder: ${frame_decoder}

      table:
        path: data/NuScenes-v1.0-mini-scene-0103.mcap
        builder:
          _target_: rbyte.io.table.TableBuilder
          _convert_: all
          reader:
            _target_: rbyte.io.table.mcap.McapTableReader
            _recursive_: false
            decoder_factories:
              - mcap_protobuf.decoder.DecoderFactory
              - rbyte.utils.mcap.McapJsonDecoderFactory
            fields:
              /CAM_FRONT/image_rect_compressed:
                log_time:
                  _target_: polars.Datetime
                  time_unit: ns
                idx: null

              /CAM_FRONT_LEFT/image_rect_compressed:
                log_time:
                  _target_: polars.Datetime
                  time_unit: ns
                idx: null

              /CAM_FRONT_RIGHT/image_rect_compressed:
                log_time:
                  _target_: polars.Datetime
                  time_unit: ns
                idx: null

              /odom:
                log_time:
                  _target_: polars.Datetime
                  time_unit: ns
                vel.x: null

          merger:
            _target_: rbyte.io.table.TableMerger
            separator: /
            merge:
              /CAM_FRONT/image_rect_compressed:
                log_time:
                  method: ref

              /CAM_FRONT_LEFT/image_rect_compressed:
                log_time:
                  method: ref
                idx:
                  method: asof
                  tolerance: 10ms
                  strategy: nearest

              /CAM_FRONT_RIGHT/image_rect_compressed:
                log_time:
                  method: ref
                idx:
                  method: asof
                  tolerance: 10ms
                  strategy: nearest

              /odom:
                log_time:
                  method: ref
                vel.x:
                  method: interp

          filter: |
            `/odom/vel.x` >= 8.6

          cache: !!null

  sample_builder:
    _target_: rbyte.sample.builder.GreedySampleTableBuilder
    index_column: /CAM_FRONT/image_rect_compressed/idx

frame_decoder:
  _target_: simplejpeg.decode_jpeg
  _partial_: true
  colorspace: rgb
  fastdct: true
  fastupsample: true
```

Build a dataloader and print a batch:
```python
from omegaconf import OmegaConf
from hydra.utils import instantiate


config = OmegaConf.load("config.yaml")
dataloader = instantiate(config.dataloader)
batch = next(iter(dataloader))
print(batch)
```

Inspect the batch:
```python
Batch(
    frame=TensorDict(
        fields={
            /CAM_FRONT/image_rect_compressed: Tensor(shape=torch.Size([1, 1, 900, 1600, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
            /CAM_FRONT_LEFT/image_rect_compressed: Tensor(shape=torch.Size([1, 1, 900, 1600, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
            /CAM_FRONT_RIGHT/image_rect_compressed: Tensor(shape=torch.Size([1, 1, 900, 1600, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
        batch_size=torch.Size([1]),
        device=None,
        is_shared=False),
    meta=BatchMeta(
        input_id=NonTensorData(data=['NuScenes-v1.0-mini-scene-0103'], batch_size=torch.Size([1]), device=None),
        sample_idx=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False),
        batch_size=torch.Size([1]),
        device=None,
        is_shared=False),
    table=TensorDict(
        fields={
            /CAM_FRONT/image_rect_compressed/idx: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
            /CAM_FRONT/image_rect_compressed/log_time: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
            /CAM_FRONT_LEFT/image_rect_compressed/idx: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
            /CAM_FRONT_LEFT/image_rect_compressed/log_time: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
            /CAM_FRONT_RIGHT/image_rect_compressed/idx: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
            /CAM_FRONT_RIGHT/image_rect_compressed/log_time: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
            /odom/vel.x: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.float64, is_shared=False)},
        batch_size=torch.Size([1]),
        device=None,
        is_shared=False),
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False)
```

Append a `logger` to `config.yaml`:
```yaml
logger:
  _target_: rbyte.viz.loggers.RerunLogger
  schema:
    frame:
      /CAM_FRONT/image_rect_compressed: 
        rerun.components.ImageBufferBatch:
          color_model: RGB

      /CAM_FRONT_LEFT/image_rect_compressed:
        rerun.components.ImageBufferBatch:
          color_model: RGB

      /CAM_FRONT_RIGHT/image_rect_compressed:
        rerun.components.ImageBufferBatch:
          color_model: RGB

    table:
      /CAM_FRONT/image_rect_compressed/log_time: rerun.TimeNanosColumn
      /CAM_FRONT/image_rect_compressed/idx: rerun.TimeSequenceColumn
      /CAM_FRONT_LEFT/image_rect_compressed/idx: rerun.TimeSequenceColumn
      /CAM_FRONT_RIGHT/image_rect_compressed/idx: rerun.TimeSequenceColumn
      /odom/vel.x: rerun.components.ScalarBatch
```

Visualize the dataset:
```python
from omegaconf import OmegaConf
from hydra.utils import instantiate


config = OmegaConf.load("config.yaml")
dataloader = instantiate(config.dataloader)
logger = instantiate(config.logger)

for batch_idx, batch in enumerate(dataloader):
    logger.log(batch_idx, batch)
```
<img alt="rerun" src="https://github.com/user-attachments/assets/c29965a7-787a-46fa-a5a0-a51f0133e5ba">

</details>

## Development

1. Install required tools:
- [`uv`](https://github.com/astral-sh/uv)
- [`just`](https://github.com/casey/just)

2. Clone:
```bash
git clone https://github.com/yaak-ai/rbyte
```

3. Setup:
```shell
just setup
```
