# Vehicle Re-Identification 🚗
Hello! This is a coursework repository for **EEEM071: Advanced Topics in Computer Vision and Deep Learning**. This is a fork of the [original course work repository](https://github.com/Surrey-EEEM071-CVDL/EEEM071-Coursework-2025) with some QoL improvements:
- Weights & Biases Integration 📊 for seamless experiment tracking
- Makefile support 🛠️ for running the experiments
- Poetry 📜 for dependency management

If you are a **University of Surrey student**, you are welcome to use this project as a learning resource and reference for your coursework. A simple credit to the OC (wee! that's me, [Frank](https://frankcholula.notion.site/)) would be greatly appreciated. However, please note that submitting this work as your own academic assignment is not permitted and may lead to [academic misconduct penalties](https://www.surrey.ac.uk/office-student-complaints-appeals-and-regulation/academic-misconduct-and-appeals). Just make sure you're submitting your orignal work.

## Code Overview 🧑‍💻
```mermaid
flowchart LR
    A[Data Loading] --> B[Model Setup]
    B --> C[Training]
    C --> D[Evaluation]
    
    subgraph A[Data Loading]
        A1[src.datasets.veri] -- data augmentation --> A2[src.transforms]
        A2 -- create train, test dataloaders --> A3["src.data_manager.return_dataloaders()"]
    end
    
    subgraph B[Model Setup]
        B1["src.models.init_model()"] -- setup xent and htri losses --> B2[src.losses]
        B2 -- configure optimizer and scheduler --> B3[src.optimizers + src.lr_schedulers]
    end
    
    subgraph C[Training]
        C1["optimizer.step()"] -- calculate losses --> C2[xent_loss + htri_losss]
        C2 -- backpropagation --> C3["loss.backward()"]
        C3 -- forward pass --> C1
    end
    
    subgraph D[Evaluation]
        D1["qf, q_pids, q_cam_ids, gf, g_pids, g_cam_ids"] -- calculate distance matrix --> D2[distmat]
        D2 -- Compute cmc and mAP --> D3["evaluate(distmat, q_pids, g_pids, q_camids, g_camids)"]
    end
```

## Directory Structure 🌳
```
VeRi
├── Makefile
├── README.md
├── args.py
├── assets
├── docs
├── experiments
│   ├── extra
│   ├── section1
│   ├── section2
│   └── section3
├── main.py
├── poetry.lock
├── pyproject.toml
├── src
│   ├── data_manager.py
│   ├── dataset_loader.py
│   ├── datasets
│   ├── eval_metrics.py
│   ├── losses
│   ├── lr_schedulers.py
│   ├── models
│   ├── optimizers.py
│   ├── samplers.py
│   ├── transforms.py
│   └── utils
└── train.sh
```
All relevant papers are stored in the `docs` directory. A sample training script is provided in `train.sh`.  I'm also using `sourcetrail` to visualize the code structure. It's been discontinued but still works fine. You can download it [here](https://github.com/CoatiSoftware/Sourcetrail).

## Running the Experiments 🏃
You can find the experiments for different sections under the `experiments` folder. Please run it using the `Makefile` in the root directory.
```bash
make section1
```
These experiments are run on my own **4080 Super FE**, with the resutls both stored locally and on [W&B](https://wandb.ai/site/).


## Experiment Results 📈
All experiment results are stored on this [W&B](https://wandb.ai/tsufanglu/VeRi?nw=nwusertsufanglu) workspace.
The summary of the best configurations for each section is as follows:

| Section     | Hyperparameter       | Best Configuration                        | mAP (%)| Rank-1 (%)|
|-------------|----------------------|-------------------------------------------|--------|-----------|
| Section 1   | Network architecture | resnet50_fc512                            |  54.54 |   87.01   |
| Section 2   | Data augmentation    | Random2DTransation + RandomHorizontalFlip |  54.54 |   87.01   |
| Section 3.1 | Learning rate        | 0.0001                                    |  64.96 |   91.06   |
| Section 3.2 | Batch size           | 64                                        |  65.85 |   91.95   |
| Section 3.3 | Optimizer            | AMSGrad                                   |  65.85 |   91.95   |

You can also find detailed results on my W&B report [here](https://api.wandb.ai/links/tsufanglu/kyjaxf8c).

## Checklist 📝
- [x] Section 1: Different network architectures
- [x] Section 2: Data augmentation
- [x] Section 3: Hypyerparameter tuning
- [x] Section 4: Summary

## License 📃
This project is licensed under the MIT License. This means:
- **Attribution**: You must give appropriate credit to the original author (me, Frank Lü) if you use or modify this project.
- **Freedom**: You are free to use, modify, and distribute this project, including for commercial purposes, without restriction.
- **No Warranty**: This project is provided “as-is” without any warranty or liability on the part of the author.
You can read the full license in the LICENSE file.
