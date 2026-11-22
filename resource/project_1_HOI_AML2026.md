## Estimating Poses of Hands Interacting with Objects Using Transformers

TA: Stephany Ortuño

Academic year 2026-27

## Task Overview

Reference paper: HaMeR: Reconstructing hands in 3d with transformers [9]. [URL 🔗](#page-0)

Driven by the rapid evolution of deep learning over the past years, Hand-Object Interaction (HOI) has matured into a field of research with applications in video understanding, AR/VR and robotics. Capturing how humans interact with their environment requires moving beyond static visual recognition toward modeling the dynamic physical mechanics of manipulation. While early computer vision paradigms [9, 13] treated hand pose estimation and object tracking as separate tasks, recent work [5, 4, 3, 12, 10, 6] favors joint estimation of hand and object pose, geometry, and contact state from visual inputs. [URL 🔗](#page-0)

Despite recent progress, estimating HOI poses from RGB images presents severe challenges due to mutual occlusions between the hand and the object and depth ambiguities. Furthermore, recovering physically plausible predictions requires preventing interpenetration between hand mesh models (such as MANO [11]) and object surfaces while maintaining physical realism. [URL 🔗](#page-0)

## Literature

Before starting the project, it is essential to review the existing literature on HOI pose estimation and reconstruction to gain a solid understanding of the concepts, challenges, and state-of-the-art techniques in the field. In the reference paper [9] and in the references section, you will find some works related to these topics to start with. Inspect the references of this paper to enter in more detail of the methods. [URL 🔗](#page-0)

- Task Distinction and Taxonomy: Clarify the functional and algorithmic differences between tasks like HOI Pose Estimation and HOI Reconstruction.

- Representation Analysis: Analyze parametric and implicit representations for hands (joint models against mesh models).

- Evaluation Metrics: Study the primary metrics used to benchmark performance across HOI datasets, including Mean Per-Joint Position Error (MPJPE), Procrustes-Aligned MPJPE (PA-MPJPE), etc.

- Loss Function Formulations: Examine key loss functions and regularization terms used on the literature.

## Dataset

The DexYCB [2] dataset is a large-scale dataset published by NVIDIA and the University of Washington in 2021. It contains around 600K multi-view RGB-D frames of hands interacting [URL 🔗](#page-0)


with 20 daily objects from the YCB dataset. Including annotations of the segmentation masks, 6D object poses, MANO [11] hand poses and 2D/3D hand joint coordinates, as shown in Figure 1. [URL 🔗](#page-0)

*Figure 1: DexYCB dataset overview.*

Dataset page: https://dex-ycb.github.io/ [URL 🔗](https://dex-ycb.github.io/)

In this project we will use just a fraction of the dataset, corresponding to a 10% of the original dataset. You can download the corresponding files in the next link. [URL 🔗](https://dex-ycb.github.io/)

[Project Dataset: https://drive.google.com/file/d/1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk](https://drive.google.com/file/d/1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk)

## Dataset Splitting

Training Set: 80%

Validation Set: 10%

Test Set: 10%

## Project Goals

## First Mandatory Part: Data inspection

This phase focuses on understanding the structure of the dataset, implementing efficient data

loading routines, and preparing the data for model training. Explore the dataset and recognize the representation of the information needed for the project (e.g joint and mesh representation). This will help you to have a deep understanding of the data and to use it in a clear way for your future model. First, create a dataset class to load the images and annotations of the dataset. Inspect the dataset by plotting the information of some of the batches. You will obtain something similar to the Figure 2. The RGB and the Depth frames are available but also the translation and rotation of each object in the scene. The joints of the hands in 2D and 3D are available too. [URL 🔗](#page-0)

## Active Object Definition

The dataset contains annotations for all objects present in a scene. Because this project targets Hand-Object Interaction (HOI), an active object is strictly defined as an object in direct contact with the hand in a given frame. Frames without active contact must be identified and handled accordingly.

## Data Augmentation

To prevent overfitting when training on a reduced dataset size, implement visual data aug- mentation techniques during training, such as Color Jitter and Gaussian Blur.


*Figure 2: DexYCB dataset inspection.*

## Second Mandatory Part: Run HaMeR in DexYCB dataset

To validate hand pose accuracy, initial experiments focus on using the pre-trained HaMeR [9] model directly to predict hand poses on the new dataset and evaluate the corresponding 3D hand position metrics (e.g. MPVPE, RVV). A second round of experiments consist of training HaMeR architecture using DexYCB [2] dataset. Report results for both experiments in the following table. Analyze the results and discuss what could be the main advantages of the pretrained or the full-trained version of the model. [URL 🔗](#page-0)

*Table 1: HaMeR performance comparison.*

| Model | MPVPE RRV PAV AUCV MPJPE RRJ PAJ AUCJ |
| --- | --- |
|   | [mm] [mm] [mm] (0–50mm) [mm] [mm] [mm] (0–50mm) |
| HaMeR-pretrained |   |
| HaMeR |   |

## Third Mandatory Part: HOI Pose Estimation Model

Once the hand pose estimation performance is established, the model architecture is extended to handle full hand-object interaction. Extend the pre-trained HaMeR model by reusing its ViT backbone as a spatial feature extractor for hand-centered RGB image crops, as shown in Figure 3. Pass input images through the frozen encoder to extract spatial feature maps, and aggregate these representations into a compact global feature vector via adaptive average pooling and a shared projection layer. [URL 🔗](#page-0)

Incorporate specialized prediction heads operating on top of the shared feature space to esti- mate both hand and object parameters. Incorporate two multi-layer perceptron (MLP) heads for the hand: one head to predict parametric MANO [11] hand pose parameters and global trans- lation, and a second head to regress relative 3D coordinates for key hand joints. Additionally, concatenate the shared latent features with an object category embedding and pass the combined features through an object pose head to predict 3D translation and normalized unit quaternions for 3D rotation. [URL 🔗](#page-0)

## Loss Function

The model optimization relies on a multi-task loss function using Mean Squared Error (MSE) across predicted and ground-truth parameters, supervising 3D hand translations, 3D object translations and rotation and relative hand-object translations.


*Figure 3: HOI Model.*

## Metrics

During evaluation, performance is evaluated using both 3D joint and surface vertex tracking metrics measured in millimeters (mm). For keypoint joint evaluation, implement and evaluate MPJPE, RRJ, PAJ (PA-MPJPE), and AUCJ across all 21 hand joints. Correspondingly, sur- face mesh fidelity must be evaluated across all 778 MANO [11] hand vertices by implementing MPVPE, RRV, PAV, and AUCV. [URL 🔗](#page-0)

In the following table, report the results obtained from HaMeR [9] on the DexYCB [2] dataset and the results obtained by your model. Note that the reported baseline values are the minimum values that you have to achieve from your model. [URL 🔗](#page-0)

*Table 2: Performance comparison across different models and evaluation metrics.*

| Model | MPVPE RRV PAV AUCV MPJPE RRJ PAJ AUCJ ADD PD |
| --- | --- |
|   | [mm] [mm] [mm] (0–50mm) [mm] [mm] [mm] (0–50mm) [mm] [mm] |
| HaMeR-pretrained | NA NA |
| Baseline | 72.82 64.51 8.80 0.17 57.62 21.20 11.02 0.11 96.90 2.42 |
| Your model |   |

Analyze the results and report what are the main sources of errors (e.g., hand pose estimation or object pose estimation). By examining the metrics determine what part of the pose is being predicting better (translation, rotation, etc) and which one could be improved.

## Qualitative Results

To visually evaluate and analyze the model performance, overlay the predicted 3D pose di- rectly onto the original RGB input images using camera intrinsic calibration parameters. Specif- ically, predicted 3D hand keypoints should be projected onto a 2D pixel space. For dense surface representations, the predicted MANO [11] 3D hand mesh (778 vertices) and the transformed YCB [1] 3D object mesh (oriented via the predicted quaternion and translation) should be rendered with dynamic transparency (e.g., α = 0.65) and distinct color overlays over the image frame. Qualitative comparisons could display ground-truth annotations alongside model predictions as shown in Figure 4 to visually diagnose pose misalignment, hand-object mutual occlusions, depth estimation errors, and geometric interpenetration. [URL 🔗](#page-0)

## Extensions

In this final phase, you will expand the baseline framework by implementing an advanced feature extension designed to improve model generalization and capability. You can choose to execute one of the suggested extensions or propose an alternative research direction. If you want to


*Figure 4: Visualizations of predicted poses.*

propose a new extension, it should be discussed and approved by the Teaching Assistant prior to implementation.

## Contact Pose Estimation

Reference paper: Learning dense hand contact estimation from imbalanced data [8]. [URL 🔗](#page-0)

The suggested extension focuses on predicting HOI contact maps directly from image ob- servations to complement pose estimation. Contact maps can be generated by computing the Euclidean point-to-surface distance between the hand and the active object’s transformed 3D tri- angle mesh, thresholding vertices within a close proximity (e.g., 5 mm) as active contact points, as shown in Figure 5. [URL 🔗](#page-0)

*Figure 5: Visualizations contact pose estimation using DexYCB dataset.*

Architecturally, this can be implemented by keeping the pre-trained HOI backbone and pose prediction heads frozen while attaching a lightweight, multi-layer perceptron (MLP) contact head. This new head consumes the shared features from the base transformer backbone and outputs logits corresponding to the per-vertex contact probability across the 778 hand vertices. To train this module, binary cross-entropy can be applied. Report the following metrics, as contact prediction is framed as a per-vertex binary classification task.

Provide a diagram with the details of the architecture and describe the parameters that you have to define to create the contact maps.


*Table 3: Performance evaluation across classification and segmentation metrics.*

| Model | Precision Recall F1-Score IoU |
| --- | --- |
| ContactNet |   |

## Multi-view Pose Estimation

Reference paper: Multi-view Hand Reconstruction with a Point-Embedded Transformer [13]. Monocular 3D HOI pose estimation suffers from depth ambiguity and visual occlusion, especially when the hand obscures key object features. While the baseline captures relative object geome- try (baseline PAV = 9.93 mm), global translation and rotation estimates in camera space remain noisy (baseline MPVPE = 72.82 mm, baseline ADD = 96.90 mm). Fusing feature representations across DexYCB’s synchronized camera views could help to resolve the depth ambiguity through cross-view geometric consistency and fills in occluded visual regions [13]. Address the global translation and alignment errors in monocular object pose estimation by leveraging DexYCB’s multi-camera architecture. Extend the current pipeline to ingest synchronized multi-view images alongside their camera extrinsics. Benchmark this extension against the single-view baseline to quantify reductions in global translation error. [URL 🔗](#page-0)

*Figure 6: Example of multi-view images included in DexYCB dataset.*

*Table 4: Performance comparison across different models and evaluation metrics.*

| Model | MPVPE RRV PAV AUCV MPJPE RRJ PAJ AUCJ ADD PD |
| --- | --- |
|   | [mm] [mm] [mm] (0–50mm) [mm] [mm] [mm] (0–50mm) [mm] [mm] |
| HaMeR-pretrained | NA NA |
| Baseline | 72.82 64.51 8.80 0.17 57.62 21.20 11.02 0.11 96.90 2.42 |
| Monocular |   |
| Multi-view |   |

By looking at the results, describe the metrics that increase the performance and discuss the reason ans how you integrate the multi-view version of the model. Provide a diagram with the details of the architecture.


## Knowledge Distillation

Reference paper: Fast-HaMeR: Boosting Hand Mesh Reconstruction using Knowledge Distillation [7]. Train your model by distilling knowledge from a larger teacher model using feature-level and output-level distillation losses. Report the metrics in comparison with your original model. Include a table to report, how does the distilled student model compare to the teacher in terms of parameter count and FLOPs. Provide a diagram of the pipeline and a detailed description of the method used to train the [URL 🔗](#page-0)

model.

## At the end

Deliver the PyTorch code for all the steps of the project, with the notebooks or scripts needed to reproduce your results. Complete and submit the report describing your results and including the required discussions.

## References

- [1] Berk Calli, Arjun Singh, AaronWalsman, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M Dollar. The ycb object and model set: Towards common benchmarks for manipulation research. In 2015 international conference on advanced robotics (ICAR), pages 510–517. IEEE, 2015.

- [2] Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, et al. Dexycb: A benchmark for capturing hand grasping of objects. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9040–9049. IEEE, 2021.

- [3] Zicong Fan, Maria Parelli, Maria Eleni Kadoglou, Xu Chen, Muhammed Kocabas, Michael J Black, and Otmar Hilliges. Hold: Category-agnostic 3d reconstruction of interacting hands and objects from video. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 494–504. IEEE, 2024.

- [4] Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. Honnotate: A method for 3d annotation of hand and object poses. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3193–3203. IEEE, 2020.

- [5] Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J Black, Ivan Laptev, and Cordelia Schmid. Learning joint reconstruction of hands and manipulated objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11807–11816, 2019.

- [6] Andela Ilic, Rachel Schuchert, Yijing Jiang, and Christian Holz. Egophi: Estimating 3d hand-object contact and force from egocentric vision. In European Conference on Computer Vision, pages 410–428. Springer, 2026.

- [7] Hunain Ahmed Jillani, Ahmed Tawfik Aboukhadra, Ahmed Elhayek, Jameel Malik, Na- dia Robertini, and Didier Stricker. Fast-hamer: Boosting hand mesh reconstruction using knowledge distillation. arXiv preprint arXiv:2603.16444, 2026.

- [8] Daniel Jung and Kyoung Mu Lee. Learning dense hand contact estimation from imbalanced data. Advances in Neural Information Processing Systems, 38:120351–120384, 2026.


- [9] Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing hands in 3d with transformers. In 2024 IEEE/CVF Confer- ence on Computer Vision and Pattern Recognition (CVPR), pages 9826–9836. IEEE, 2024.

- [10] Patrick Rim, Kevin Harris, Braden Copple, Shangchen Han, Xu Xie, Ivan Shugurov, Sizhe An, He Wen, Alex Wong, Tomas Hodan, et al. Show3d: Capturing scenes of 3d hands and objects in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7111–7120, 2026.

- [11] Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: Modeling and capturing hands and bodies together. arXiv preprint arXiv:2201.02610, 2022.

- [12] Anilkumar Swamy, Vincent Leroy, Philippe Weinzaepfel, Fabien Baradel, Salma Galaaoui, Romain Brégier, Matthieu Armando, Jean-Sebastien Franco, and Grégory Rogez. Showme: Benchmarking object-agnostic hand-object 3d reconstruction. In 2023 IEEE/CVF Inter- national Conference on Computer Vision Workshops (ICCVW), pages 1927–1936. IEEE, 2023.

- [13] Lixin Yang, Licheng Zhong, Pengxiang Zhu, Xinyu Zhan, Junxiao Kong, Jian Xu, and Cewu Lu. Multi-view hand reconstruction with a point-embedded transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
