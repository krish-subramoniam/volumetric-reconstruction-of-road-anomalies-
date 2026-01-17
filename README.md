# Metric-Accurate Volumetric Reconstruction of Road Anomalies  
### Using Binocular Stereo Vision (Traditional Computer Vision)

**Course:** 22AIE313 – Computer Vision & Image Processing  
**Mentor:** Sajith Variyar V. V.  
**Focus Area:** Traditional Computer Vision (No Deep Learning)

---

##  Project Overview

This project presents a **purely geometric computer vision pipeline** for detecting and **quantifying the volume of road anomalies** such as **potholes and road humps** using **binocular stereo vision**.

Unlike many modern approaches that rely on deep learning, this work strictly adheres to **first principles of image formation, stereo correspondence, epipolar geometry, and 3D reconstruction**, as prescribed in the **22AIE313 syllabus**.

The final output of the system is a **volumetric estimate (in cm³ / liters)** of road damage, enabling direct **civil-engineering relevance**, such as estimating the quantity of filling material required for repair.

---

## 🎯 Key Objectives

- Perform dense stereo correspondence without neural networks  
- Reconstruct 3D road geometry using calibrated stereo vision  
- Detect potholes and humps via geometric deviation from ground plane  
- Quantify anomaly volume using 3D metrology  
- Demonstrate a complete **end-to-end traditional CV pipeline**

---

##  Methodology Pipeline

The system follows a four-stage pipeline:

1. **Camera Calibration (Unit 1)**  
   - Based on Zhang’s checkerboard method (assumed/approximated for dataset)
   - Intrinsic parameters (focal length, principal point) used for metric depth

2. **Stereo Correspondence (Unit 2 & Unit 3)**  
   - Dense disparity estimation using **Semi-Global Block Matching (SGBM)**
   - Enforces epipolar constraints

3. **3D Reconstruction (Unit 3)**  
   - Disparity → depth using stereo geometry  
   - Reprojection to 3D using the Q-matrix  
   - Depth values clamped to realistic road ranges (0.5 m – 30 m)

4. **3D Metrology & Volumetric Quantification**  
   - Ground plane estimation via **RANSAC**
   - Potholes and humps detected as deviations from plane
   - Volume computed by integrating depth deviation over surface area

---

##  Project Structure
There are 11 stereo image pairs ,totalling upto 22 images
Metric_Pothole_3D/
│
├── data/
│ └── dataset1/
│ └── rgb/
│ ├── 000001_left.png
│ ├── 000001_right.png
│ ├── 000002_left.png
│ ├── 000002_right.png
│ └── ...
│
├── pothole_volume_pipeline.py
├── README.md

