# Real-time Semantic Segmentation for Mapillary Vistas Dataset

## White Paper
![Thumbnail](https://github.com/ComputerFish/Hybrid-Attention-Segformer/blob/4696b7f17458f00924a1b33d62a85e4a9a7bf564/white%20paper/Thumbnail.jpg)
[PDF Copy](https://github.com/ComputerFish/Hybrid-Attention-Segformer/blob/088abc1ac44fd6ae115ed7680561ee7cea801010/white%20paper/Hybrid-%20SegFormer.pdf)

The paper introduces a hybrid attention module that fuses local and global token dependencies.

## Project Structure

```bash
.
├── dataset_preprocessed_512x512
│   ├── config_v1.2.json
│   ├── training
│   │   ├── images
│   │   │   ├── -0C1J9CvgFP4BTVLXNeNZA.jpg
│   │   │   └── ...
│   │   └── labels
│   │       ├── -0C1J9CvgFP4BTVLXNeNZA.png
│   │       └── ...
│   └── validation
│       ├── images
│       │   ├── -3-MmXdwhyIQhtb4-8NqHQ.jpg
│       │   └── ...
│       └── labels
│           ├── -3-MmXdwhyIQhtb4-8NqHQ.png
│           └── ...
│
├── trainer
│   ├── config.py
│   ├── dataset.py
│   ├── main.py
│   ├── model.py
│   ├── train.py
│   └── windowedLocalSelfAttention.py
│
├── outputs
│   ├── Segformer_B2
│   │   ├── model.pt
│   │   ├── predictions
│   │   │   ├── -3-MmXdwhyIQhtb4-8NqHQ
│   │   │   │   ├── image.jpg
│   │   │   │   ├── mask.png
│   │   │   │   └── pred.png
│   │   │   └── ...
│   │   └── training_log.csv
│   └── ...
│
└── demo
    ├── demo_config.py
    └── demo.py
```

---

## How to Download Dataset

This project uses a custom pre-processed version of the Mapillary Vistas Dataset. All images are initially resized to a height of 512 px, with the width scaled proportionally. They are then center-cropped to ensure all final input images are a square size of 512x512 pixels.

1.  **Install `gdown`:**
    ```bash
    pip3 install gdown
    ```

2.  **Download the Dataset:**
    ```bash
    gdown 11ZPJbu9ZVcWOaSFcyV6tUnwuW107nrM3
    ```

3.  **Unzip the File:**
    ```bash
    unzip dataset_preprocessed_512x512.zip
    ```

4.  **Place into Structure:**
    Place the extracted `dataset_preprocessed_512x512` directory in the project root, matching the structure detailed in the Project Structure section.

---

## How to Train the Model

This section contains instruction on how to train the model. In this project, we evaluated five distinct configurations based on the SegFormer backbone.

### Prerequisites

Before starting training, ensure all necessary dependencies are installed and that the dataset is correctly placed in the `dataset_preprocessed_512x512` directory.

### Training Execution

1.  **Change Directory:**
    ```bash
    cd trainer
    ```

2.  **Run Training Command:**
    Choose one of the following commands to start training the desired model architecture. The training parameters (e.g. epochs, batch size) are managed within `config.py` but can be overriden with command line arguments (e.g. --epochs).

    ```bash
    # SegFormer Baseline Models
    python3 main.py --model Segformer_B0
    python3 main.py --model Segformer_B2

    # Modified Models (with Windowed Local Self-Attention)
    python3 main.py --model Segformer_B0_modified_1
    python3 main.py --model Segformer_B0_modified_2
    python3 main.py --model Segformer_B2_modified_1
    ```

### Output

* The training process will automatically save checkpoints and logs.
* Results will be placed in the **`../outputs/`** directory, categorized by model name (e.g. `../outputs/Segformer_B2/`).
* The trained model weights are saved as **`model.pt`** within the corresponding model folder.
* The trained model will also be used to predict mask of validation images. The result is saved in the `predictions` directory.

---

## How to Run Live Demo

The live demonstration uses the `demo.py` script to perform real-time segmentation using one of the trained models.

### Prerequisites

* **Trained Model:** Ensure you have successfully completed training the model to be used (`model.pt`) is available in your `outputs/` directory.

### Configuration Options

To change to a different model for real-time segmentation, you can specify it in the demo_config.py or override it in the command line arguments:

* The `--model` specifies which trained model architecture to load (e.g., `Segformer_B2`, etc.).
* The `--model_filepath` specifies the filepath to saved trained model (`model.pt`).

### Execution

1.  **Change Directory:**
    ```bash
    cd demo
    ```

2.  **Run the Demo:**
    ```bash
    python3 demo.py
    ```