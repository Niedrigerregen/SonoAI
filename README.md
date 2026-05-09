----
SonoAI
----
A TensorFlow and Keras-based model for detecting fetal brain tumors in ultrasound images.

----
Overview
----
SonoAI is an experimental deep learning project exploring whether Convolutional Neural Networks (CNNs) can effectively learn from unevenly distributed datasets in a highly complex medical imaging domain — fetal brain tumor detection.

This project aims to push the limits of CNN generalization and augmentation strategies when data scarcity and imbalance are major challenges.

----
Requirements for recreating
----
- Python 3.10 - 3.12
- A CUDA(basically just Nvidia) supporting GPU(for GPU training)
- This [Dataset](https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset) or any other medical Dataset
----
Project Goals
----
- Investigate the performance of CNNs on imbalanced medical datasets.
- Explore data augmentation and regularization techniques to improve generalization.
- Evaluate how model architecture depth and callback strategies affect training stability.
- Assess whether meaningful predictions can still emerge from limited, noisy ultrasound data.

----
Model Details:
----
- Frameworks: TensorFlow & Keras
- Core Model: Custom CNN architecture with multiple convolutional and pooling layers
- 400k Parameters
- Batch size of 32
- 128 filters max.
- max. 1000 epochs (Early stopping)
- Used ~100 epochs until early_stopping(CPU)

----
Techniques Used:
----

	* Early Stopping
	* Data augmentation (rotation, zoom, flip, etc.)
	* Dropout and batch normalization
	* Custom callbacks for model monitoring
	* manually duplicating the pictures in the folder by flipping them

----
Results & Observations:
----
	* The model demonstrates surprisingly good learning capability even with such data.
	* with a testaccuracy of ~ 80% - 85% it's far from being good enough to be used in medical contexts
	* Heavy augmentation, a fairly big droputrate and careful tuning are essential to avoid overfitting.
	*

----
Future Work:
----
	* Add the Benign Class to the model (extending to Multiclass Classification)
	* Add real-time class recognition
	* Make a GUI version and convert it into an .exe file

----
Disclaimer:
----
This is my first Project working with AI therefore results won't be very good.

I did not create the dataset required for the CNN.

This project is not intended for any use besides research.

----
Dataset Attribution:
----

This project uses the Ultrasound Fetus Dataset available on [Kaggle](https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset).
The dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
You are free to share and adapt the material for any purpose, even commercially, as long as appropriate credit is given.

Citation
Anitha, A (2024). Ultrasound Fetus Dataset. Mendeley Data, V1, doi: 10.17632/yrzzw9m6kk.1

----
*made by Loran Hisso, not AI*
----
