# CSIKit-BMI

This is the official implementation for _BMEye: Public Health-Oriented Body Mass Index Monitoring using Commodity WiFi_ accepted at WF-IoT '23
Heatmaps in the dataset are 2-D images converted using code in Section CSI-Heatmap conversion.

[Paper](https://drive.google.com/file/d/1uPdt7CdH3Zn_0uXA3-Ol2WQ02ktm_pxr/view?usp=drive_link)

![System_overview](https://github.com/kiran-collab/CSIKit-BMI/assets/75129341/3f5247ee-4578-4411-a27e-04a50d17cf70)

Train given CSI Heatmaps:
1. The dataset folder contains individual class Heatmaps in .zip files. Unzip them and place them there only.
2. Run 'resnet50-csi-pytorch.py' to train with above data.

Train custom CSI Samples: 
1. To train on custom CSI data, convert CSI .pcap/.dat files using code in Section CSI-Heatmap conversion.
