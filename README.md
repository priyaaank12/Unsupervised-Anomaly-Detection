# Unsupervised Anomaly Detection in Industrial Inspection

A working computer vision model that learns what a "normal" (non-defective) product
looks like, and flags anything that deviates from that as an anomaly (scratch, dent,
crack, contamination, missing part, etc.) — **without ever being shown a labeled
defect image during training.**

---

## 1. Folder Structure

```
anomaly_detection_project/
│
├── data/
│   ├── train/
│   │   └── good/              # ONLY normal/defect-free images go here
│   └── test/
│       ├── good/              # normal test images
│       └── defective/         # defective test images (for evaluation only)
│
├── models/
│   └── autoencoder_best.pth   # saved trained model weights (created after training)
│
├── src/
│   ├── model.py                # the neural network (Convolutional Autoencoder)
│   ├── dataset.py               # loads & preprocesses images
│   ├── train.py                  # training script
│   ├── evaluate.py               # computes anomaly scores + heatmaps + metrics
│   └── utils.py                    # helper functions
│
├── app/
│   └── app.py                       # Streamlit web app to deploy & test the model live
│
├── outputs/
│   └── (heatmaps, ROC curve, sample results get saved here)
│
├── requirements.txt
└── README.md
```

## 2. Which Dataset to Use

The industry-standard benchmark for this exact project is **MVTec AD**
(MVTec Anomaly Detection dataset) — free for research/academic use.
It has categories like `bottle`, `cable`, `capsule`, `screw`, `metal_nut`, `hazelnut`, etc.,
each with:
- `train/good/` → only normal images
- `test/good/` + `test/<defect_type>/` → normal + various defects
- `ground_truth/` → pixel-level defect masks (optional, for advanced scoring)

Download: https://www.mvtec.com/company/research/datasets/mvtec-ad

Just pick **one category** (e.g. `bottle` or `hazelnut` — small and fast to train),
and drop its folders into `data/train/good` and `data/test/good` + `data/test/defective`
matching the structure above.

If you don't want to use MVTec AD, you can use **your own product photos** — just make
sure `train/good` has only clean/normal images (50–300 images is enough to start).

## 3. The Roadmap (Big Picture)

| Phase | What you do | File |
|---|---|---|
| 1. Data prep | Organize images into good/defective folders, resize, normalize | `dataset.py` |
| 2. Model design | Build a Convolutional Autoencoder (CAE) | `model.py` |
| 3. Training | Train CAE on ONLY normal images to minimize reconstruction loss | `train.py` |
| 4. Thresholding | Compute reconstruction error on validation "good" images to set an anomaly threshold | `evaluate.py` |
| 5. Evaluation | Test on good + defective images, compute AUC-ROC, F1, accuracy, view heatmaps | `evaluate.py` |
| 6. Deployment | Wrap model in a Streamlit web app for live image upload + anomaly detection | `app/app.py` |
| 7. (Optional upgrade) | Swap CAE for a stronger method: PaDiM / PatchCore / Anomalib | — |

