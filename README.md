# Vehicle Re-Identification 🚗
Hello! This is a coursework repository for **EEEM071: Advanced Topics in Computer Vision and Deep Learning**. This is a fork of the [original course work repository](https://github.com/Surrey-EEEM071-CVDL/EEEM071-Coursework-2025) with some QoL improvements:
- Weights & Biases Integration 📊 for seamless experiment tracking
- Makefile support 🛠️ for running the experiments
- Poetry 📜 for dependency management

If you are a **University of Surrey student**, you are welcome to use this project as a learning resource and reference for your coursework. A simple credit to the OC (wee! that's me, [Frank](https://frankcholula.notion.site/)) would be greatly appreciated. However, please note that submitting this work as your own academic assignment is not permitted and may lead to [academic misconduct penalties](https://www.surrey.ac.uk/office-student-complaints-appeals-and-regulation/academic-misconduct-and-appeals). Just make sure you're submitting your orignal work.


## Code Structure
```
VeRi
├── Makefile
├── README.md
├── __pycache__
├── args.py
├── docs
├── experiments
├── logs
├── main.py
├── poetry.lock
├── pyproject.toml
├── src
├── train.sh
└── wandb
```
All relevant papers are stored in the `docs` directory. A sample training script is provided in `train.sh`.  I'm also using `sourcetrail` to visualize the code structure. It's been discontinued but still works fine. You can download it [here](https://github.com/CoatiSoftware/Sourcetrail).

## Running the Experiments 🏃
You can find the experiments for different sections under the `experiments` folder. Please run it using the `Makefile` in the root directory.
```bash
make section1
```
These experiments are run on my own **4080 Super FE**, with the resutls both stored locally and on [W&B](https://wandb.ai/site/).


## License 📃
This project is licensed under the MIT License. This means:
- **Attribution**: You must give appropriate credit to the original author (me, Frank Lü) if you use or modify this project.
- **Freedom**: You are free to use, modify, and distribute this project, including for commercial purposes, without restriction.
- **No Warranty**: This project is provided “as-is” without any warranty or liability on the part of the author.
You can read the full license in the LICENSE file.
