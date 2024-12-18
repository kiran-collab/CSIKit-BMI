# CSIKit-BMI

This is the official implementation for _BMEye: Public Health-Oriented Body Mass Index Monitoring using Commodity WiFi_ accepted at WF-IoT '23.

Heatmaps in the dataset are 2-D images converted using code in Section CSI-Heatmap conversion.

[[Paper]](https://drive.google.com/file/d/1uPdt7CdH3Zn_0uXA3-Ol2WQ02ktm_pxr/view?usp=drive_link)

![System_overview](https://github.com/kiran-collab/CSIKit-BMI/assets/75129341/3f5247ee-4578-4411-a27e-04a50d17cf70)

# 🔧 Requirements

Requirements are given in requirements.txt file 

To install all packages and libraries needed, run _pip install -r requirements.txt_

# ⚡ CSI- Heatmap Conversion

# ⚡ Train given CSI Heatmaps:
1. The dataset folder contains individual class Heatmaps in .zip files. Unzip them and place them there only.
2. Run 'resnet50-csi-pytorch.py' to train with above data.

# ⚡ Train custom CSI Samples: 
1. To train on custom CSI data, convert CSI .pcap/.dat files using code in 'CSI-Heatmap conversion' section.

If you use this work, please consider citing it:

@INPROCEEDINGS{10539515,
  author={Davuluri, Kiran and Mottakin, Khairul and Song, Zheng and Lu, Jin and Allison, Mark},
  booktitle={2023 IEEE 9th World Forum on Internet of Things (WF-IoT)}, 
  title={BMEye: Public Health-Oriented Body Mass Index Monitoring Using Commodity WiFi}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  keywords={Weight measurement;Shape;Sociology;Stability criteria;Sensors;Statistics;Public healthcare;Channel State Information (CSI);WiFi Sensing;Machine Learning;BMI Classification;Public Health},
  doi={10.1109/WF-IoT58464.2023.10539515}}
