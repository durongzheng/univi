在开发视觉模型的过程中，我们使用过很多的视觉检测模型，比如 yolo, swin transformer, vit 等。我们需要实验各种模型在工作中的实际表现。
在众多的视觉模型中， Ultralytics 封装的 Yolo 代码库是目前封装得最为完善、优雅，同时能够兼容各种不同视觉任务的一套代码，包括分类，检测，分割，姿态估计，方向识别等任务。  
但是，视觉模型是在不断演化进步的，有一些当前最优的模型并没有在yolo的代码中得以实现，比如: swinv2；非互斥的多标签分类也没有实现。另外，很多功能对于本地训练和部署的模型而言，又完全没有必要。因此，我们用 Ultralytics 的代码构建了一套最小化的框架代码，以简化并统一自身的实验过程。  
代码几乎全部来自 Ultralytics, 也与其遵循同样的 AGPL-3.0 协议。

安装：

```bash
git clone https://github.com/durongzheng/univi.git
cd univi
conda create -n univi python=3.10 -y
conda activate univi
pip install .
```

</details>

<details open>
<summary>Usage</summary>

### CLI

YOLOv8 may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` can be used for a variety of tasks and modes and accepts additional arguments, i.e. `imgsz=640`. See the YOLOv8 [CLI Docs](https://docs.ultralytics.com/usage/cli) for examples.

### Python

YOLOv8 may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from univi import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco8.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.


## <div align="center">License</div>

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/licenses/) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://ultralytics.com/license).

