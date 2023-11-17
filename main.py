import cv2
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from streamlit_drawable_canvas import st_canvas


import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255.0
X_test_flattened = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255.0

# Make the model_epochs variable global
model_epochs = None


# Function to create and train a neural network
def train_neural_network(activation, optimizer, loss, epochs):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(10, activation=activation)
    ])

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    history = model.fit(X_train_flattened, y_train, epochs=epochs, verbose=0)

    return model, history


# Function to preprocess the drawn image

def preprocess_drawn_image(image_data):
    # Convert RGBA image to grayscale
    img = image_data.astype(np.uint8)
    img = img[:, :, :3]  # Exclude the alpha channel
    img = np.mean(img, axis=-1)  # Convert to grayscale

    # Invert colors
    img = 255 - img

    # Reshape and normalize
    img = img.reshape(1, -1) / 255.0

    # Ensure the correct input shape
    img = img[:, :784]  # Trim or pad if necessary

    return img


# Introduction to how computers "see"
def computer_vision_intro():
    st.title("Introduction to Image Classification")

    st.markdown(
        """
Hey there!  Imagine you have a magical friend who can look at pictures and understand them, just like you do. Computers are a bit like that magical friend. 
They use special tricks with numbers to understand pictures, and we're going to have some fun exploring how they do it!
Now, we have a special set of pictures called "handwritten digits." These are like the numbers you write on paper, but they are made up of tiny dots.
Each dot has a number that tells the computer how dark or light it is. Our mission is to teach the computer to recognize and understand these handwritten digits. 
Think of it like teaching your friend to recognize different shapes and colors. The computer will learn from lots of examples, just like you learn by seeing things many times.
Once it learns, it can look at a new handwritten number and tell you what it is! It's a bit like how you can recognize your friend's face after seeing it many times.
Now, here's the exciting part: we can teach the computer in different ways! We can change how it learns and how it looks at the pictures. It's like giving your friend new tools to recognize things better. 
Let's have a blast exploring these cool ways and training our own computer vision model! 
        """
    )

    st.markdown("### How Computers See: Example Image")

    # Display a single example image
    st.image("/Users/kapilsharma/Desktop/Kapil/Vizuara/Assignment/Image.png", caption="How Computers see", width=800)

    st.markdown(
        """
Imagine you have this fantastic collection of pictures with dogs and cats. Now, when you look at these pictures, you notice some cool things. 
Dogs, for example, often have longer snouts and floppy ears, while cats have pointy ears and slender faces. It's like a little secret code that helps you tell them apart.
Now, when you mark each picture with a label (dog or cat), your computer friend gets curious. It wants to know your secret code! 
So, you tell it, "Look, dogs usually have longer snouts and floppy ears, and cats have pointy ears and slender faces."
Your computer friend then puts on its special learning glasses (we call it a machine learning model). It looks at all the pictures you labeled and starts noticing these little details: the shapes of ears, the length of snouts, and more.
During this learning time, the computer friend becomes really good at spotting these differences. It's like a superhero with a superpower: "Spot the Dog" or "Detect the Cat."
Now comes the exciting part - making predictions! When you show the computer a new picture, it starts looking for those secret codes. 
Does it see a longer snout? Floppy ears? It might confidently say, "Ah, I think this is a dog!" Or if it spots pointy ears and a slender face, it might declare, "This looks like a cat!"
It's like the computer is a detective too, using the clues you shared to crack the case of "Is it a dog or a cat?" And just like you, it gets better and better at this detective work as it sees more and more pictures.
So, the next time you show your computer friend a picture, remember it's using its detective skills to find those little details and make its best guess - whether it's a furry dog or a sleek cat!
        """
    )


# MNIST Dataset Information
def mnist_dataset():
    st.title("Create your own Dataset")

    st.markdown(
        """
        ## MNIST Dataset

Imagine you have a special album full of pictures, but these pictures are a bit different. They are not just regular photos; they are pictures of handwritten numbers from 0 to 9.
It's like a magical collection of numbers created by people writing them down.

Now, each of these numbers is like a tiny drawing made up of small dots, just like connecting the dots in a coloring book. But here's the cool part: each dot has its own special number.
When we look at these dots, we can see shades of gray, from bright white to deep black. The number 0 means it's super bright, and 255 means it's really dark.
So, our goal is to teach our computer friend to recognize these handwritten numbers. We call this magical collection the MNIST dataset. 
It's like a training ground for the computer to become a superhero in understanding and reading these handwritten digits.
The MNIST database is like a treasure trove, filled with 60,000 pictures for practice and 10,000 pictures for testing our computer's skills. 
These pictures were collected and put together to help computers become really good at recognizing numbers. It's like a school for computers to learn the art of understanding handwriting.
Here's a fun fact: The MNIST database was created by mixing samples from NIST's original datasets. NIST stands for the National Institute of Standards and Technology, and they had some fantastic datasets that became the foundation for our magical MNIST collection.

Now, when the computer is trained using this special dataset, it gets so good that it can recognize these handwritten numbers with an error rate of just 0.8%! That's like making only a tiny mistake out of every hundred guesses.
It's like having a super-smart friend who can look at any handwritten number and tell you exactly what it is.

So, the MNIST dataset is like a playground where computers learn the art of reading handwritten numbers, and they become really, really good at it!

        ### Visualize Sample

        Use the slider below to select the number of samples you want to visualize.
        """
    )

    # Slider to select the number of samples
    num_samples = st.slider("Number of Samples to Visualize", min_value=1, max_value=10, value=5)

    # Visualize samples
    fig, axs = plt.subplots(1, num_samples, figsize=(12, 3))
    for i in range(num_samples):
        axs[i].imshow(X_train[i], cmap='gray')
        axs[i].axis('off')

    st.pyplot(fig)

    # Train-Test Split
    st.markdown(
        """
        ### Train-Test Split
        
Imagine you have a box of yummy cookies, and you want to make sure they taste just right. Now, to check how delicious they are, you don't eat all the cookies at once, right? 
You save some for later to see if they still taste good.
The Train-Test Split is a bit like that. When we have a bunch of special pictures (let's call them our "dataset"), and we want to teach the computer how to recognize things in them, we use a similar idea.

**Number of Data Samples**: Think of the number of data samples as how many cookies you want to taste. If you have lots of cookies, you might want to taste only a few to make sure the rest are just as yummy.

**Split Ratio**: Now, the split ratio is like deciding how many cookies you'll eat now (the "Train" part) and how many you'll save for later to check if your taste buds are still happy (the "Test" part).
It's like saying, "I'll eat 80% of the cookies now and save 20% for later."
So, when we're dealing with our special pictures, the computer looks at some of them (the "Train" part) to learn and become really good at recognizing things.
Then, it takes what it learned and tests itself on the pictures it hasn't seen before (the "Test" part) to make sure it's super smart.
Just like you taste a few cookies to make sure the whole batch is delicious, the computer checks a part of the pictures to make sure it understands all of them.
This way, we know the computer is really good at recognizing things in any picture, not just the ones it practiced with.
So, Train-Test Split is like a yummy adventure – learning from some pictures and making sure that learning works on other pictures too!

Use the sliders below to generate a train-test split of the dataset. Specify the number of data samples and the split ratio.
        """
    )

    # Sliders for Train-Test Split
    num_data_samples = st.slider("Number of Data Samples", min_value=100, max_value=len(X_train), value=500)
    split_ratio = st.slider("Train-Test Split Ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.1)

    # Generate Train-Test split
    num_train_samples = int(num_data_samples * split_ratio)
    num_test_samples = num_data_samples - num_train_samples

    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train_flattened[:num_data_samples],
        y_train[:num_data_samples],
        test_size=num_test_samples,
        random_state=42
    )

    st.write(f"Number of Train Samples: {num_train_samples}")
    st.write(f"Number of Test Samples: {num_test_samples}")


def model():
    st.title("Build your own model")

    st.markdown(
        """
Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold.

**Artificial Neuron**: It is the most basic and primitive form of any neural network. It is a computational unit which performs the following steps
1. It takes certain inputs and weights.
2. Applies dot product on respective inputs & weights and apply summation.
3. Apply some transformation using activation function on the above summation.
4. Fires output.

There are many activation functions that apply different types of transformations to incoming signals in the neuron. Activation functions are necessary to bring non-linearity in the neural network

**Input Layer**: This layer consists of the input data which is being given to the neural network.
This layer is depicted like neurons only but they are not the actual artificial neuron with computational capabilitie
Each neuron represents a feature of the data. This means if we have a data set with three attributes Age, Salary, City then we will have 3 neurons in the input layer to represent each of them. If we are working with an image of the dimension of 1024×768 pixels then we will have 1024*768 = 786432 neurons in the input layer to represent each of the pixels          

**Hidden Layer**: This is the layer that consists of the actual artificial neurons. If the number of hidden layer is one then it is known as a shallow neural network.
If the number of hidden layer is more than one then it is known as a deep neural network. In a deep neural network, the output of neurons in one hidden layer is the input to the next hidden layer.          
    """
    )

    st.image("/Users/kapilsharma/Desktop/Kapil/Vizuara/Assignment/ANN.GIF",
             caption="Neural Network Animation",
             width=800)

    st.markdown(
        """
**Output Layer**: This layer is used to represent the output of the neural network.
The number of output neurons depends on number of output that we are expecting in the problem at hand.  

**Weights and Bias**: The neurons in the neural network are connected to each other by weights.
Apart from weights, each neuron also has its own bias.     

**Backpropagation**: During the training phase, the neural network is initialized with random weight values. Training data is fed to the network and the network then calculates the output. This is known as a forward pass. The calculated output is then compared with the actual output with the help of loss/cost function and the error is determined.
Now comes the backpropagation part where the network determines how to adjust all the weights in its network so that the loss can be minimized. This weight adjustment starts happening from the rear end of the network. The error is propagated in the backward direction to the front layers till the end and the neurons across the network start adjusting their weights. Hence the name backpropagation.  
"""
    )

    st.markdown("### How Neural network works")

    # Display a single example image

    st.image("/Users/kapilsharma/Desktop/Kapil/Vizuara/Assignment/New animation.GIF",
             caption="Neural Network Animation",
             width=800)


# Neural Network Trainer
def neural_network_trainer():
    global model_epochs  # Access the global model_epochs variable

    st.title("Neural Network Trainer")

    st.markdown(
        """
        Customize your neural network training by selecting different parameters.
        """
    )

    # Option 1: Vary Activation Function
    st.header("Vary Activation Function")
    activation_functions = ['sigmoid', 'relu', 'tanh','elu', 'selu']
    selected_activation = st.selectbox("Select Activation Function", activation_functions)

    # Train the model with Adam optimizer, sparse categorical cross entropy, and 5 epochs
    model_epochs, history_activation = train_neural_network(selected_activation, 'adam',
                                                            'sparse_categorical_crossentropy', 5)

    # Evaluate the model
    loss_activation, accuracy_activation = model_epochs.evaluate(X_test_flattened, y_test)
    st.write(f"Model Accuracy with {selected_activation} activation: {accuracy_activation * 100:.2f}%")
    st.write(f"Model Loss: {loss_activation:.4f}")

    # Plot accuracy over epochs for activation variation
    st.subheader("Effect of Activation function Variation by keeping constant optimizer,epochs and loss")
    fig_activation = plt.figure()
    plt.plot(history_activation.history['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy with {selected_activation} Activation')
    st.pyplot(fig_activation)

    # Option 2: Vary Number of Epochs
    st.header("Effect of Varying Number of Epochs by keeping all other paramters constant")
    epochs_range = st.slider("Select Number of Epochs", min_value=1, max_value=10, value=5)

    # Train the model with Adam optimizer, sparse categorical cross entropy, and varying epochs
    model_epochs, history_epochs = train_neural_network('softmax', 'adam', 'sparse_categorical_crossentropy', epochs_range)

    # Evaluate the model
    loss_epochs, accuracy_epochs = model_epochs.evaluate(X_test_flattened, y_test)
    st.write(f"Model Accuracy with {epochs_range} epochs: {accuracy_epochs * 100:.2f}%")
    st.write(f"Model Loss: {loss_epochs:.4f}")

    # Plot accuracy over epochs for epoch variation
    st.subheader("Accuracy Over Epochs (Epoch Variation)")
    fig_epochs = plt.figure()
    plt.plot(history_epochs.history['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Over Epochs with {epochs_range} Epochs')
    st.pyplot(fig_epochs)


# Function to make predictions
def make_predictions():
    def predictDigit(image):
        model = tf.keras.models.load_model(
            r"/Users/kapilsharma/Desktop/Kapil/Vizuara/Sid/Handwritten-Digit-Recognition-main/model/handwritten.h5")
        image = ImageOps.grayscale(image)
        img = image.resize((28, 28))
        img = np.array(img, dtype='float32')
        img = img / 255
        plt.imshow(img)
        plt.show()
        img = img.reshape((1, 28, 28, 1))
        pred = model.predict(img)
        result = np.argmax(pred[0])
        return result

    # Streamlit
  #  st.set_page_config(page_title='Handwritten Digit Recognition', layout='wide')
    st.title('Handwritten Digit Recognition')
    st.subheader("Draw the digit on canvas and click on 'Predict Now'")

    # Add canvas component
    # Specify canvas parameters in application
    drawing_mode = "freedraw"
    stroke_width = st.slider('Select Stroke Width', 1, 30, 15)
    stroke_color = '#FFFFFF'  # Set background color to white
    bg_color = '#000000'

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=200,
        width=200,
        key="canvas",
    )

    # Add "Predict Now" button
    if st.button('Predict Now'):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
            input_image.save('img.png')
            img = Image.open("img.png")
            res = predictDigit(img)
            st.header('Predicted Digit: ' + str(res))
        else:
            st.header('Please draw a digit on the canvas.')


# Streamlit App
def main():
    global model_epochs  # Access the global model_epochs variable

    st.sidebar.title("Navigation")
    pages = {
        "Introduction": computer_vision_intro,
        "MNIST Dataset": mnist_dataset,
        "Build your own model": model,
        "Neural Network Trainer": neural_network_trainer,
        "Digit Recognizer": make_predictions
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()


if __name__ == "__main__":
    main()

