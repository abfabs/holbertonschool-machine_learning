![Transfer Learning Illustration](https://images.unsplash.com/photo-1555949963-aa79dcee981c)

# Transfer Learning for CIFAR-10 Image Classification

> "Research is what I'm doing when I don't know what I'm doing." — Wernher von Braun

---

## Abstract

In this experiment, I trained a convolutional neural network capable of classifying images from the CIFAR-10 dataset with a validation accuracy above 87%. Instead of training a deep neural network from scratch, I applied **transfer learning** using a pretrained model from the Keras Applications module.

The experiment consisted of adapting a pretrained convolutional architecture to the CIFAR-10 dataset by resizing the input images, freezing most pretrained layers, training a custom classification head, and then fine-tuning a subset of the deeper layers.

This approach significantly reduced training time while achieving strong performance on a relatively small dataset. The final model reached a validation accuracy above the required threshold and was saved as a compiled model (`cifar10.h5`) for later evaluation.

---

## Introduction

Image classification is a fundamental task in computer vision. It involves assigning a label to an image based on its visual content. Modern convolutional neural networks (CNNs) have achieved remarkable performance on large datasets such as ImageNet, but training such models from scratch requires massive computational resources and enormous labeled datasets.

The **CIFAR-10 dataset** is a widely used benchmark in machine learning. It consists of:

- 60,000 images across **10 classes**
- 32×32 RGB images
- 50,000 training images and 10,000 testing images

The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

While CIFAR-10 is relatively small compared to datasets like ImageNet, training a deep CNN from scratch still requires careful architecture design and long training times. To solve this more efficiently, I used **transfer learning** — a technique where knowledge learned from one task is reused to solve another related task — by leveraging pretrained weights from a network trained on ImageNet and adapting it for CIFAR-10 classification.

---

## Materials and Methods

### Dataset

The CIFAR-10 dataset was loaded using Keras:

```python
(X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
```

Each image has a shape of `(32, 32, 3)`. However, most pretrained models in Keras are trained on larger images such as 224×224 or 96×96, so resizing was required.

### Data Preprocessing

Two preprocessing steps were performed:

**1. Image normalization**

The dataset was normalized using the preprocessing function associated with the pretrained model:

```python
X_p = K.applications.mobilenet_v2.preprocess_input(X.astype("float32"))
```

This scales pixel values into the range expected by the pretrained network.

**2. One-hot encoding of labels**

Labels were converted to categorical format:

```python
Y_p = K.utils.to_categorical(Y.reshape(-1), 10)
```

This converts a label such as `3` into a vector like `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`, which neural networks can use for classification.

### Model Architecture

The model was built using **MobileNetV2**, a lightweight convolutional network designed for efficient computation. The architecture consisted of three main components:

**1. Input resizing layer**

Since CIFAR-10 images are 32×32, they were resized using a Lambda layer:

```python
Lambda(lambda img: K.layers.Resizing(96, 96)(img))
```

**2. Pretrained feature extractor**

The MobileNetV2 base model was loaded with ImageNet weights:

```python
base_model = K.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(96, 96, 3)
)
```

The majority of these layers were frozen, meaning their weights were not updated during training. This allows the model to reuse visual features learned from millions of images.

**3. Custom classification head**

A new classifier was added on top of the pretrained features:

```python
x = GlobalAveragePooling2D()(features)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(10, activation="softmax")(x)
```

This head learns how to map the extracted visual features to the 10 CIFAR-10 classes.

### Training Strategy

Training was performed in two phases:

**Phase 1 — Training the classifier head**

All pretrained layers were frozen and only the classification head was trained. To speed things up, the outputs of the frozen layers were precomputed once and reused:

```python
train_features = feature_extractor.predict(X_train)
```

This dramatically reduced training time since the expensive convolutional layers did not need to be recomputed during every epoch.

**Phase 2 — Fine-tuning**

After the classifier head converged, the final portion of the pretrained network was unfrozen and fine-tuned using a very small learning rate:

```python
learning_rate = 1e-5
```

This allowed the network to adapt high-level features to CIFAR-10 without destroying the pretrained knowledge.

---

## Results

The final model achieved the following performance on the test set:

| Metric        | Value  |
|---------------|--------|
| Test Accuracy | 0.8864 |
| Loss          | 0.3329 |

This exceeded the required 87% validation accuracy threshold. Key observations:

- Transfer learning significantly improved baseline performance
- Freezing pretrained layers dramatically reduced training time
- Fine-tuning further improved accuracy
- Data preprocessing was essential for compatibility with pretrained models

The trained model was saved as `cifar10.h5`.

---

## Discussion

This experiment demonstrates the effectiveness of transfer learning in computer vision tasks. Instead of designing and training a deep CNN from scratch, pretrained models allow researchers to leverage knowledge learned from extremely large datasets. Even though CIFAR-10 images are small and differ from ImageNet images, the pretrained features still provide a strong foundation for classification.

Several factors contributed to the model's success: proper preprocessing, input resizing, freezing pretrained layers, training a custom classifier, and gradual fine-tuning. Without transfer learning, achieving similar accuracy would require a more complex architecture and significantly longer training times.

This workflow reflects a common practice in modern machine learning: **reuse pretrained models whenever possible.**

---

## Acknowledgments

- TensorFlow / Keras framework
- CIFAR-10 dataset creators
- Keras Applications pretrained models
- *Densely Connected Convolutional Networks* (Huang et al., 2017)

---

## Literature Cited

1. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.*
2. Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.*
3. Huang, G. et al. (2017). *Densely Connected Convolutional Networks.*
4. TensorFlow Documentation — [https://www.tensorflow.org/](https://www.tensorflow.org/)
5. Keras Applications Documentation — [https://keras.io/api/applications/](https://keras.io/api/applications/)