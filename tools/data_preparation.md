# Dataset Preparation
You need to create a new conda environment to do these preparatory works.

**Step 1: Generate mask of the object shape.**

* Clone [RemoteSAM](https://github.com/1e12Leon/RemoteSAM) repo.

* Move [extract_classnames.py](./extract_classnames.py) and [start_seg.py](./start_seg.py)  to the folder.

* Generate the classnames of each images and save to the output_dir by python extract_classnames.py.

* Generate the target mask image for the labels and save  to the output_dir by python start_seg.py.

  (After experiments, we found that for the data that has been pre-trained in RemoteSAM, the effect of directly extracting the category information from labelTxt for semantic segmentation is basically the same as that of extracting the caption through RemoteCLIP and then performing instance segmentation. Therefore, we provide more convenient semantic segmentation code to reduce the intermediate steps.)

**Step 2: Generate prompt.json.**

* As shown in Figure 8 of the paper, captions can affect the bias of the model's generation. When not using reinforcement learning strategies, we recommend not using captions to improve the evaluation metrics of the model's generation, including the performance improvement of downstream tasks and the distribution of images. 
* Here, we provide two scripts, namely [generate_wcap_json.py](./data/generate_wcap_json.py) and [generate_wocap_json.py](./data/generate_wocap_json.py) .
* Generate the prompt.json by python ./data/generate_wocap_json.py or python ./data/generate_wcap_json.py.

The above steps can complete the basic training and inference of the model. We will release other data processing codes soon.

