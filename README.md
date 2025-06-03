# Yoga Pose Classifier

A deep learning project that classifies yoga poses using Convolutional Neural Networks (CNN). The model can identify and classify 5 different yoga poses: Downdog, Goddess, Plank, Tree, and Warrior2.

## 🧘‍♀️ Project Overview

This project uses TensorFlow and Keras to build a CNN model that can accurately classify yoga poses from images. The model achieves high accuracy in distinguishing between the 5 different yoga poses included in the dataset.

## 🎯 Supported Yoga Poses

1. **Downdog** - Downward-Facing Dog pose
2. **Goddess** - Goddess pose  
3. **Plank** - Plank pose
4. **Tree** - Tree pose
5. **Warrior2** - Warrior II pose

## 🚀 Features

- CNN-based image classification
- Data augmentation for better generalization
- Model training with validation
- Real-time pose prediction
- Pre-trained model included (model.h5)

## 📁 Project Structure

```
yoga-pose-classifier/
├── DATASET/
│   ├── TRAIN/          # Training images organized by pose
│   │   ├── downdog/
│   │   ├── goddess/
│   │   ├── plank/
│   │   ├── tree/
│   │   └── warrior2/
│   └── TEST/           # Test images organized by pose
│       ├── downdog/
│       ├── goddess/
│       ├── plank/
│       ├── tree/
│       └── warrior2/
├── yoga_pose12s.ipynb  # Main training notebook
├── Main.py             # Inference script
├── model.h5            # Pre-trained model
├── training_logs.csv   # Training history
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/yoga-pose-classifier.git
   cd yoga-pose-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python Main.py
   ```

## 📊 Dataset

The project uses a yoga poses dataset with:
- **Training set**: 1,081 images across 5 classes
- **Test set**: 470 images across 5 classes
- **Image size**: 150x150 pixels
- **Classes**: 5 different yoga poses

*Dataset source*: [Yoga Poses Dataset on Kaggle](https://www.kaggle.com/niharika41298/yoga-poses-dataset)

## 🔥 Usage

### Training the Model

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook yoga_pose12s.ipynb
   ```

2. Run all cells to train the model from scratch

### Making Predictions

1. **Using the inference script**:
   ```bash
   python Main.py
   ```

2. **Using the model programmatically**:
   ```python
   import tensorflow as tf
   import cv2
   import numpy as np
   
   # Load the model
   model = tf.keras.models.load_model('model.h5')
   
   # Load and preprocess image
   image = cv2.imread('path/to/your/image.jpg')
   image = cv2.resize(image, (150, 150))
   image = np.array(image).reshape(1, 150, 150, 3)
   
   # Make prediction
   prediction = model.predict(image)[0]
   
   # Convert to class name
   poses = {1: 'Downdog', 2: 'Goddess', 3: 'Plank', 4: 'Tree', 5: 'Warrior2'}
   predicted_pose = poses[np.argmax(prediction) + 1]
   print(f'Predicted pose: {predicted_pose}')
   ```

## 🏗️ Model Architecture

The CNN model consists of:
- **Input Layer**: 150x150x3 (RGB images)
- **Convolutional Blocks**: 4 blocks with increasing filter sizes (64, 128, 256, 256)
- **Pooling**: MaxPooling after each block
- **Dropout**: 0.2 dropout rate for regularization
- **Dense Layers**: Fully connected layers for classification
- **Output**: 5 classes with softmax activation

## 📈 Model Performance

- **Training Accuracy**: High accuracy achieved during training
- **Validation Accuracy**: Good generalization on test set
- **Training History**: Available in `training_logs.csv`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Anuj Dev Singh**
- Project Creator & Lead Developer

## 🙏 Acknowledgments

- Dataset provided by [Niharika41298 on Kaggle](https://www.kaggle.com/niharika41298/yoga-poses-dataset)   
- Built with TensorFlow and Keras
- OpenCV for image processing

## 📞 Contact

If you have any questions or suggestions, feel free to reach out!

---

**Happy Coding! 🧘‍♂️✨** 
