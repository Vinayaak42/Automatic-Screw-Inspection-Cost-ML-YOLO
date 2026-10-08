# 🔩 Automatic Screw Inspection Using Machine Learning & Vision Control

### AI-Based Quality Inspection and Cost Reduction for Mobile Assembly Stations

An industrial **Computer Vision + Machine Learning** solution designed to automatically inspect screw installation in mobile assembly stations.

The system detects screw-related defects such as **missing screws, incorrect screw position, misalignment, wrong screw type, and damaged screw heads**, then generates an automated **PASS / FAIL** decision.

The primary objective is not only defect detection, but also **manufacturing cost reduction** by reducing manual inspection effort, minimizing defect escapes, reducing rework/scrap, and improving process monitoring.

---

## 📌 Project Overview

In manufacturing assembly lines, screw inspection is often performed manually or through partially automated inspection methods.

Manual inspection can introduce:

* Operator fatigue
* Inspection time
* Human error
* Inconsistent inspection decisions
* Missed defects
* Rework
* Scrap
* Production delays
* Additional quality inspection cost

This project proposes an automated vision-based inspection system that captures assembly images, detects screws, evaluates their condition and position, and produces an automated quality decision.

```text
Mobile Assembly Station
          │
          ▼
     Camera Capture
          │
          ▼
   Image Preprocessing
          │
          ▼
    Screw Detection
          │
          ▼
 Position / Orientation
      Verification
          │
          ▼
   Defect Classification
          │
          ▼
     PASS / FAIL
          │
      ┌───┴────┐
      ▼        ▼
    PASS      FAIL
      │        │
 Continue    Reject /
 Assembly    Rework
      │        │
      └────┬───┘
           ▼
     Quality Database
           │
           ▼
     Analytics Dashboard
```

---

# 🎯 Business Problem

Screw installation is a critical step in many assembly processes.

A missing, misplaced, damaged, or incorrectly installed screw can result in:

* Product quality issues
* Rework
* Scrap
* Line stoppage
* Customer complaints
* Warranty costs
* Additional inspection effort
* Production delays

Traditional manual inspection requires operators to repeatedly verify screw positions across multiple products.

The business problem can therefore be represented as:

```text
Manual Inspection
       ↓
Inspection Time
       ↓
Labor Effort
       ↓
Inspection Cost

Missed Defect
       ↓
Defect Escape
       ↓
Rework / Scrap
       ↓
Additional Cost
```

The proposed system aims to break this cycle through automated inspection.

---

# 💰 Cost-Saving Objective

The main objective of this project is:

> **Use Computer Vision and Machine Learning to automate screw inspection and reduce the overall cost associated with manual inspection, defect escapes, rework, and scrap.**

The project focuses on four major cost drivers:

### 1. Manual Inspection Cost

Automating repetitive visual inspection can reduce the amount of operator time required for screw verification.

### 2. Rework Cost

Early detection of incorrect screw installation can prevent defective assemblies from moving to later production stages.

### 3. Scrap Cost

Detecting defects at the assembly station can reduce the probability of defective products reaching final inspection.

### 4. Quality Inspection Cost

Automated inspection can provide consistent inspection decisions while continuously generating inspection data for analysis.

---

# 📊 Cost-Saving Framework

The project can quantify potential savings using:

```text
Annual Manual Inspection Cost
                -
Annual Automated Inspection Cost
                +
Avoided Rework Cost
                +
Avoided Scrap Cost
                =
Potential Annual Cost Saving
```

A more detailed model:

```text
Manual Inspection Cost
=
Inspection Hours × Labor Cost / Hour

Rework Cost
=
Number of Reworked Units × Cost per Rework

Scrap Cost
=
Number of Scrapped Units × Cost per Unit

Total Current Quality Cost
=
Inspection Cost
+ Rework Cost
+ Scrap Cost
```

After automation:

```text
Automation Cost
=
System Maintenance
+ Hardware
+ Software
+ Monitoring

Potential Saving
=
Current Quality Cost - Automation Cost
```

> **Important:** Financial savings should be calculated using actual production data. This repository does not assume savings figures without validated manufacturing data.

---

# 🚀 Project Objectives

## Primary Objectives

* Automate screw inspection using Computer Vision.
* Detect screw presence and position.
* Identify defective screw conditions.
* Classify assemblies as PASS or FAIL.
* Reduce manual inspection effort.
* Reduce defect escape probability.
* Reduce rework and scrap.
* Track inspection performance.
* Monitor defect trends.
* Track ML experiments using MLflow.

## Secondary Objectives

* Build a reusable industrial inspection pipeline.
* Create a quality analytics dashboard.
* Store inspection results for further analysis.
* Enable model performance monitoring.
* Provide a foundation for future PLC/industrial camera integration.

---

# 🔍 Defects Covered

The system can be designed to identify:

| Defect                  | Description                                       |
| ----------------------- | ------------------------------------------------- |
| ✅ PASS                  | Correct screw and correct position                |
| ❌ Missing Screw         | Expected screw is absent                          |
| ❌ Misaligned Screw      | Screw is outside the acceptable position          |
| ❌ Wrong Screw           | Incorrect screw type detected                     |
| ❌ Damaged Screw         | Screw/head appears damaged                        |
| ❌ Incorrect Position    | Screw does not match expected location            |
| ❌ Multiple Screw        | Unexpected additional screw                       |
| ❌ Incorrect Orientation | Screw orientation differs from expected condition |

Additional defect classes can be added as the dataset grows.

---

# 🧠 Solution Architecture

```text
                    ┌─────────────────────┐
                    │ Mobile Assembly     │
                    │ Station             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Industrial Camera   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Acquisition   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ OpenCV              │
                    │ Preprocessing       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ YOLO / ML Model     │
                    │ Screw Detection     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Position & Defect   │
                    │ Analysis            │
                    └──────────┬──────────┘
                               │
                               ▼
                     ┌────────┴────────┐
                     │                 │
                   PASS              FAIL
                     │                 │
                     ▼                 ▼
                 Continue          Reject /
                 Assembly           Rework
                     │                 │
                     └────────┬────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Inspection Database │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘
```

---

# 🖼️ Computer Vision Pipeline

The image-processing pipeline consists of:

### Step 1 — Image Acquisition

Capture assembly images using a camera positioned above or around the assembly station.

### Step 2 — Image Preprocessing

Typical preprocessing operations include:

* Image resizing
* Noise reduction
* Contrast enhancement
* Brightness correction
* ROI extraction
* Image normalization

### Step 3 — Screw Detection

The ML model identifies screw locations within the assembly image.

Example:

```text
Input Image
     ↓
YOLO Detection
     ↓
┌─────────────────────────┐
│ Screw 1     Screw 2     │
│    ✓           ✓        │
│                         │
│ Screw 3     Missing     │
│    ✓          ✗         │
└─────────────────────────┘
```

### Step 4 — Position Verification

Detected screw coordinates are compared against expected screw positions.

### Step 5 — Defect Decision

The system determines whether the assembly satisfies predefined quality conditions.

```text
All Expected Screws Present
          +
Correct Position
          +
Correct Type
          +
Acceptable Confidence
          ↓
         PASS
```

Otherwise:

```text
One or More Conditions Failed
          ↓
         FAIL
```

---

# 🤖 Machine Learning Approach

The project can use an object-detection model such as **YOLO** for screw detection and classification.

The model learns from annotated assembly images.

### Input

```text
Assembly Image
```

### Output

```text
Bounding Box
Class
Confidence Score
```

Example:

```text
Class: Screw
Confidence: 0.97

Class: Missing Screw
Confidence: 0.94
```

---

# 📚 Dataset

The dataset can contain images captured from the assembly environment.

Recommended categories:

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── validation/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

The dataset should represent realistic production variation such as:

* Lighting variation
* Camera angle variation
* Product position variation
* Screw orientation
* Different backgrounds
* Minor assembly variation
* Different defect conditions

---

# 🏷️ Image Annotation

Object-detection annotations can be created using tools such as:

* Roboflow
* LabelImg
* CVAT

Example classes:

```text
0 → screw
1 → missing_screw
2 → wrong_screw
3 → misaligned_screw
4 → damaged_screw
```

The exact class structure can be modified according to the actual inspection requirement.

---

# 📈 Model Evaluation

The model should be evaluated using:

* Precision
* Recall
* F1-score
* mAP
* Confusion Matrix
* Inference Time
* False Positive Rate
* False Negative Rate

For industrial inspection, **recall is particularly important** because missed defects can result in defect escapes.

Example:

```text
Metric
-------------------------
Precision       0.XX
Recall          0.XX
F1 Score        0.XX
mAP@50          0.XX
Inference Time  XX ms
```

> Actual values will be updated after model training and validation.

---

# 🧪 MLflow Experiment Tracking

MLflow is used to track model experiments and compare different training configurations.

### Parameters

```text
Model
Image Size
Batch Size
Epochs
Learning Rate
Confidence Threshold
Data Augmentation
Optimizer
```

### Metrics

```text
Precision
Recall
F1 Score
mAP
Inference Time
```

### Artifacts

```text
Trained Model
Confusion Matrix
Prediction Images
Training Curves
Evaluation Reports
```

Example experiment structure:

```text
Screw-Inspection
│
├── Baseline Model
├── Augmented Model
├── Optimized Model
└── Production Candidate
```

This makes model development reproducible and allows data scientists to compare experiments systematically.

---

# 💡 Cost Reduction Strategy

The project connects technical ML performance with manufacturing economics.

## Cost Driver 1 — Manual Inspection

```text
Before

Operator
   ↓
Inspect Every Assembly
   ↓
Manual Decision
```

```text
After

Camera
   ↓
Automated Inspection
   ↓
ML Decision
```

Potential benefit:

**Reduced repetitive inspection effort.**

---

## Cost Driver 2 — Defect Escape

```text
Assembly Defect
      ↓
Not Detected
      ↓
Next Process
      ↓
Final Inspection
      ↓
Rework / Scrap
```

Automated inspection:

```text
Assembly Defect
      ↓
Vision Inspection
      ↓
Immediate Detection
      ↓
Correction
```

Potential benefit:

**Earlier defect detection and reduced downstream quality cost.**

---

## Cost Driver 3 — Rework

If the system detects an incorrect screw installation immediately, the product can be corrected before additional manufacturing operations are performed.

Potential benefit:

**Lower rework effort and reduced material consumption.**

---

## Cost Driver 4 — Scrap

Early detection can reduce the probability that defective assemblies proceed through expensive downstream processes.

Potential benefit:

**Reduced avoidable scrap and associated processing cost.**

---

# 📊 Quality Analytics Dashboard

A Streamlit dashboard can provide:

### KPI Cards

```text
Total Inspections
        12,540

PASS Rate
        96.8%

FAIL Rate
        3.2%

Defects Detected
        401
```

### Dashboard Sections

* Inspection volume
* PASS / FAIL rate
* Defect distribution
* Station-wise defect rate
* Hourly inspection trend
* Daily defect trend
* Top defect categories
* Model confidence
* Recent failed inspections
* Defect images
* Cost-saving indicators

---

# 📉 Cost-Saving Dashboard

A dedicated cost-analysis section can display:

```text
Estimated Manual Inspection Cost
              ₹ XX,XXX

Estimated Rework Cost Avoided
              ₹ XX,XXX

Estimated Scrap Cost Avoided
              ₹ XX,XXX

Automation Cost
              ₹ XX,XXX

Potential Net Saving
              ₹ XX,XXX

Potential Cost Reduction
              XX%
```

These values should be calculated from actual production assumptions or validated business data.

---

# 📐 Key Business KPIs

The system can track:

### Quality KPIs

```text
Defect Rate
First Pass Yield
False Reject Rate
Defect Escape Rate
Rework Rate
Scrap Rate
```

### Inspection KPIs

```text
Inspection Volume
Inspection Time
Model Inference Time
Detection Accuracy
Inspection Coverage
```

### Financial KPIs

```text
Inspection Labor Cost
Rework Cost
Scrap Cost
Cost per Defect
Cost Avoided
Potential Annual Saving
ROI
```

---

# 🧮 ROI Calculation

A simple ROI framework:

```text
Annual Benefit
=
Manual Inspection Savings
+
Rework Savings
+
Scrap Savings
```

Then:

```text
ROI
=
(Annual Benefit - Annual Automation Cost)
/
Annual Automation Cost
× 100
```

Example structure:

```text
Annual Manual Inspection Cost    ₹ A
Annual Rework Cost               ₹ B
Annual Scrap Cost                ₹ C
-------------------------------------
Current Annual Quality Cost      ₹ D

Automation Cost                  ₹ E

Potential Annual Saving
= D - E
```

Actual financial values should be calculated using production data.

---

# 🔄 End-to-End Workflow

```text
1. Product enters assembly station
              ↓
2. Camera captures image
              ↓
3. Image preprocessing
              ↓
4. Screw detection
              ↓
5. Screw position verification
              ↓
6. Defect classification
              ↓
7. PASS / FAIL decision
              ↓
8. Inspection result stored
              ↓
9. Dashboard updated
              ↓
10. Quality team monitors trends
              ↓
11. Defect data used for process improvement
```

---

# 🏭 Manufacturing Use Case

The solution can be integrated into a mobile assembly station where each product is inspected before moving to the next manufacturing stage.

Potential control logic:

```text
                 Inspection
                     │
             ┌───────┴────────┐
             │                │
           PASS              FAIL
             │                │
             ▼                ▼
       Green Signal       Red Signal
             │                │
             ▼                ▼
     Continue Process    Stop / Reject
```

For a real industrial deployment, the vision system can communicate with a PLC or manufacturing-control system.

---

# 🛠️ Technology Stack

## Programming

* Python
* SQL

## Computer Vision

* OpenCV
* Image Processing
* Object Detection

## Machine Learning

* YOLO
* Scikit-learn
* Model Evaluation

## Experiment Tracking

* MLflow

## Analytics

* Pandas
* NumPy
* Matplotlib
* Seaborn

## Dashboard

* Streamlit
* Power BI

## API / Deployment

* FastAPI
* Docker

## Data Storage

* SQLite / PostgreSQL

---

# 📂 Project Structure

```text
Automatic-Screw-Inspection-Cost-Saving-ML/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── detection.py
│   ├── classification.py
│   ├── inspection.py
│   └── utils.py
│
├── models/
│
├── mlflow/
│
├── app/
│   ├── streamlit_app.py
│   └── dashboard.py
│
├── api/
│   └── main.py
│
├── config/
│   └── config.yaml
│
├── tests/
│
├── reports/
│   ├── metrics/
│   └── predictions/
│
├── screenshots/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Automatic-Screw-Inspection-Cost-Saving-ML.git
```

Navigate into the project:

```bash
cd Automatic-Screw-Inspection-Cost-Saving-ML
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Open the displayed local URL in your browser.

---

# 🔬 Running MLflow

Start the MLflow tracking interface:

```bash
python -m mlflow ui
```

Then open the MLflow interface in your browser.

---

# 🧪 Model Training

Example:

```bash
python src/train.py
```

The training pipeline should:

```text
Load Dataset
     ↓
Preprocess Images
     ↓
Train Model
     ↓
Evaluate Model
     ↓
Log Parameters
     ↓
Log Metrics
     ↓
Log Artifacts
     ↓
Save Model
```

---

# 📊 Expected Project Outputs

The completed system should generate:

```text
✓ Trained inspection model
✓ PASS / FAIL predictions
✓ Defect classification
✓ Annotated inspection images
✓ Model evaluation metrics
✓ MLflow experiments
✓ Quality dashboard
✓ Defect analytics
✓ Cost-saving analysis
✓ Inspection history
```

---

# 💰 Business Impact

The project is designed around the following business value:

```text
Automated Inspection
        ↓
Reduced Manual Inspection Effort
        ↓
Earlier Defect Detection
        ↓
Reduced Rework
        ↓
Reduced Scrap
        ↓
Improved Quality
        ↓
Potential Cost Reduction
```

The final cost benefit should be validated using actual:

* Production volume
* Inspection time
* Labor cost
* Defect rate
* Rework cost
* Scrap cost
* Automation cost

---

# 🔮 Future Improvements

Future versions can include:

### 1. Real-Time Camera Integration

Connect the model directly to an industrial camera.

### 2. PLC Integration

Send PASS / FAIL signals directly to the assembly control system.

### 3. Edge Deployment

Deploy the model on an industrial edge device for low-latency inference.

### 4. Automatic Reject Mechanism

Trigger a pneumatic or robotic rejection mechanism for failed assemblies.

### 5. Advanced Defect Detection

Add:

* Cross-thread detection
* Loose screw detection
* Screw torque correlation
* Screw head damage detection
* Thread quality analysis

### 6. Predictive Quality

Use historical inspection data to predict defect probability before failure occurs.

### 7. Production Analytics

Integrate inspection data with manufacturing databases and Power BI.

### 8. Automated Cost Analytics

Automatically calculate:

```text
Defect Cost
Rework Cost
Scrap Cost
Inspection Cost
Cost Avoided
ROI
```

---

# 🔐 Data & Privacy

No proprietary manufacturing images or confidential production data should be uploaded to this repository.

If real factory data is used, replace it with:

* Anonymized images
* Synthetic data
* Public datasets
* Sample inspection images

before publishing the project publicly.

---

# 📌 Project Status

```text
🚧 In Development
```

Planned milestones:

```text
[ ] Dataset preparation
[ ] Image annotation
[ ] Computer vision preprocessing
[ ] Baseline detection model
[ ] YOLO model
[ ] Model evaluation
[ ] MLflow tracking
[ ] Inspection pipeline
[ ] Streamlit dashboard
[ ] Cost-saving calculator
[ ] API deployment
[ ] Final documentation
```

---

# 👨‍💻 Author

**Vinayak Kesti**

**Data Scientist | Data Analyst | AI / ML Engineer**

Focused on applying Machine Learning, Computer Vision, Data Analytics, and AI to real-world manufacturing and business problems.

---

# ⭐ Key Skills Demonstrated

```text
Python
SQL
Machine Learning
Computer Vision
Deep Learning
YOLO
OpenCV
MLflow
Data Analytics
Streamlit
Power BI
FastAPI
Model Evaluation
Quality Analytics
Manufacturing Analytics
Cost Optimization
```

---

# 📈 Project Value

This project demonstrates how an AI solution can connect:

**Computer Vision → Machine Learning → Quality Engineering → Data Analytics → Manufacturing Automation → Cost Reduction**

rather than treating machine learning as an isolated prediction problem.

---

## ⭐ If you find this project useful

Give the repository a ⭐ and feel free to explore the implementation.
