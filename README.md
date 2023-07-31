# Enhancing-By-Subtasks-Components

This project presents an experimental approach to tackle the challenge of data scarcity in a specific task by exploring the utilization of existing annotated datasets from related Nlp task. Our experiment involves training a single base model, such as BERT, with multiple heads, each dedicated to a specific task, and running them simultaneously during training. We term these additional tasks as "supporting tasks." The goal is to leverage shared knowledge across different domains and enhance the model's performance and robustness.

**Branches:**
- Medical tasks can be found in the `main` branch.
- The GLUE (General Language Understanding Evaluation) tasks can be found in the `glue_tasks` branch.

Please note that this project is experimental, and the results may vary based on the specific task and datasets used. While the approach shows promise, it is essential to interpret the outcomes with caution. The aim of sharing this experiment is to encourage collaborative exploration and discussions on dealing with data scarcity in machine learning projects.

We welcome contributions and feedback from the community to refine further and improve this experimental approach. Together, let's delve into innovative methods to overcome data limitations and advance the field of machine learning. 🌟

The multi-head model can be viewed in ```models/multiHeadModel.py```<br/>  The multi-head training can be viewed at ```train.py```

## Multi-head model architecture 
![image](https://github.com/NivAm12/Enhancing-By-Subtasks-Components/assets/49129250/00e70a4c-00de-416b-be88-3fce760f3230)

![Advanced NLP Project](https://github.com/NivAm12/Enhancing-By-Subtasks-Components/assets/68702877/d672ae7a-e7ee-4443-88d7-3b8481e225ad)


# Install
``` pip install -r requirements.txt ```

# Train
Run:

``` python train.py  --batch_size <batch size> --epochs <number of epochs> --device <device>```<br/>

For the rest of the arguments, please see ```train.py```

