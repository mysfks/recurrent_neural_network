# Sentiment Analysis with IMDB Dataset using LSTM in TensorFlow

This project focuses on creating a TensorFlow-based sentiment analysis model using the **IMDB dataset** which contains movie reviews labeled as positive or negative. The sentiment analysis task is a form of binary classification where the input is a movie review (a sequence of words) and the output is a sentiment (positive or negative).

## Project Structure

The code follows these key steps:

1. **Data Downloading and Preprocessing**:
   - We utilize the **IMDB dataset** included with TensorFlow/Keras.
   - Each review is already tokenized into integers corresponding to words, and we only use the top 10,000 most frequent words (`num_words=10000`).
   - Both train and test datasets are concatenated for further operations.
   - The dataset is divided into **Training**, **Validation**, and **Testing** datasets following the sequence: 
     - 40,000 samples for training,
     - 5,000 samples for validation, and
     - 5,000 samples for testing.
   - Padding is applied to ensure all input sequences have the same length (`maxlen=1024`).

2. **Model Architecture**:
   - The model constructed is a **Recurrent Neural Network (RNN)** with an **LSTM layer** to handle sequential data.
   - It uses an **embedding layer** to represent words in dense vectors of a fixed size (256), followed by:
     - **Dropout layers** to avoid overfitting,
     - **LSTM Layer** to capture long-term dependencies from sequential data,
     - **Dense Layer (128 units)** with RELU activation,
     - **Output Layer** with a single unit (since it’s a binary classification problem) and Sigmoid activation function.

3. **Training**:
   - The model is compiled using the **Adam optimizer** and **binary cross-entropy loss function** and is trained over 5 epochs using the training and validation datasets.
   
4. **Evaluation and Visualization**:
   - Training/Validation **Accuracy** and **Loss** plots are generated to analyze the model's learning process.
   - The final model is also evaluated on the **test dataset** to assess real-world performance.

5. **Prediction**:
   - A single test review is selected, reshaped, and fed into the trained model for prediction. The predicted sentiment and its corresponding label are printed.

## Files

- The code is written in one Python script, and the relevant plots are generated inside the script.
- There are no other extra files in this repository, apart from the TensorFlow code snippet and any dependencies you might decide to include for custom configurations.

## Pre-requisites

- **Python 3.x**: This project uses `Python 3.x`.
- **TensorFlow 2.x**: The deep learning framework. TensorFlow must be installed in your environment.
- **NumPy**: The numerical computational library, used for handling arrays.
- **Matplotlib**: The visualization library, used for plotting accuracy and loss curves.

Make sure to install these dependencies before running the code:

```bash
pip install tensorflow numpy matplotlib
```

## Workflow

### 1. Loading and Preprocessing Data
The IMDB dataset is loaded with `tf.keras.datasets.imdb.load_data()` and is processed as follows:
- Limiting vocabulary to the top 10,000 frequent words.
- Padding sequences to a length of 1024 using `tf.keras.preprocessing.sequence.pad_sequences`.

### 2. Model Architecture
The model leverages the following layers:
- **Embedding Layer**: Turns input integers (word indices) into dense vectors of fixed size.
- **LSTM Layer**: A recurrent layer to handle sequential data effectively.
- **Dense Layer**: Fully connected layer with ReLU activation for further processing after the LSTM.
- **Dropout Layers**: Used after each trainable layer to prevent overfitting by randomly setting the output to zero during training.

### 3. Training
The model is trained for 5 epochs using `model.fit()` with the validation dataset specified for monitoring overfitting.

### 4. Visualization
Two plots are generated for visualizing the training process:
- **Training and Validation Loss vs Epochs**
- **Training and Validation Accuracy vs Epochs**

### 5. Evaluation
The model is evaluated on the test set using `model.evaluate()` to understand how well it generalizes to unseen data.

### 6. Prediction
A prediction is made on a single reshaped test sample, and its corresponding actual label is compared with the model's prediction.

## Sample Output

- The code prints intermediary updates such as dataset shapes and training progress.
- After training, you will observe the following outputs:
  - Plot of **Loss vs Epochs**.
  - Plot of **Accuracy vs Epochs**.
  - Model performance during evaluation.
  - Predicted and true sentiment for a test sample.

## Results

We expect the model to reach a decent accuracy, but due to the relatively small number of epochs (5), the model might still have room for improvement. The addition of Dropout layers should reduce overfitting.

Due to the **LSTM** architecture's ability to handle sequential input, the model is effective for movie review sentiment classification.
