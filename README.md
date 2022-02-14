# Aedes Egg Detection

## Brief Description
Performing Aedes mosquito eggs detection based on HSV segmentation and image classification. 
HSV segmentation and eggs localization was done on [this repo](https://github.com/nikkopg/AedesEggLocalization) and image classification for detection was carried out based on shape classification. Shape features were extracted using [Elliptic Fourier Descriptor](http://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Kuhl-Giardina-CGIP1982.pdf) using [pyEFD library](https://pyefd.readthedocs.io/en/latest/). Common machine learning algorithms, such as Support Vector Machine, k-Nearest Neighbors, and Random Forest was trained using egg and not egg images. The 'not egg' class includes cropped egg and random shapes.
### 'not egg' class sample images
![no-12](https://user-images.githubusercontent.com/70200533/151665903-e7eb42c6-dca3-4525-861c-db60b7e266ad.jpeg)
![no-18](https://user-images.githubusercontent.com/70200533/151665904-3502d8c8-ccb9-49b2-953e-741fcdcb6dc5.jpeg)
![no-11](https://user-images.githubusercontent.com/70200533/151666126-3c4ffddd-ea8c-4b1b-9acf-02e644274037.jpeg)

### 'egg' class sample images
![yes- (18)](https://user-images.githubusercontent.com/70200533/151665939-93db9ecd-dfda-4095-a7ff-90cf617394db.jpeg)
![yes- (12)](https://user-images.githubusercontent.com/70200533/151665936-6164e480-a1ca-41ae-96e5-0b1eea68498d.jpeg)
![yes- (13)](https://user-images.githubusercontent.com/70200533/151665937-c6269b48-17f7-4572-9e19-d5fadd460726.jpeg)

Since the dataset only has 30 images of each class, data augmentation was conducted. The augmentation was done by rotating each image 360 degrees and extracting features every 5 degrees. So each image will be augmented to new 72 images with angle variation.

Hyperparameters of each model were optimized using Gridsearch and cross-validation with hyperparameter settings and search space referring to a [paper](https://arxiv.org/abs/2007.15745).

## Limitations and future improvement:
![image](https://user-images.githubusercontent.com/70200533/153872804-88d72d0e-cd6e-46fa-9fd4-8c9a624a5d8c.png)
As the model was trained using the shape of an egg (which is oval) and random/cropped egg shape (not oval), there are often found overlapping eggs that were segmented as one object and detected as 'not egg' (since it's not oval-like in the image). To overcome this, the overlapping object separation method could be used to further analyze and detect each egg that is overlapped.

## Sample output:
![Output-13](https://user-images.githubusercontent.com/70200533/151666769-3c262580-3007-47bd-8caa-b22423613aec.jpeg)
