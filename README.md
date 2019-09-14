# Hack the North 2019 - Machine Learning Workshop

Microsoft

## Overview

In this workshop, you will build an image classifier with Azure Custom Vision Service and export it as a TensorFlow model. You will learn how to import the model into a Python application and use it to classify images.

## Part 1 - Train and export the TensorFlow model

Follow the following instructions to create a Custom Vision project. You can use the training images linked to in the tutorial to build a cats vs dogs image classifier.

[**Tutorial: Train and export a TensorFlow model with Azure Custom Vision Service**](train-custom-vision-model.md)

> If you are feeling adventurous or if you have another idea in mind, you can download and use your own images to train a model that recognizes other objects.

## Part 2 - Import and use the model in Python

1. If you haven't cloned this repo already, clone it to your local machine.

1. In a terminal, browse to the `app` folder.

1. Create and activate Python 3 virtual environment by entering:

    ```bash
    python3.7 -m venv .venv
    source .venv/bin/activate
    # windows:
    # .venv\scripts\activate
    ```

1. Copy the `model.pb` and `labels.txt` that you downloaded from Azure Custom Vision Service into *model* folder or use the ones that are already there.

1. Run the script and a prediction should appear:

    ```bash
    python run.py
    ```


## Resources

Azure Custom Vision Service can export models that run on many platforms.

- [iOS and Android](https://docs.microsoft.com/azure/cognitive-services/custom-vision-service/export-your-model)
- [Xamarin (cross-platform)](https://channel9.msdn.com/Shows/XamarinShow/Custom-Vision--Object-Detection-Made-Easy)
- [Serverless HTTP API (Python)](https://docs.microsoft.com/azure/azure-functions/functions-machine-learning-tensorflow)
- You can also export it as a Docker container (Dockerfile)

Also check out these other resources:

- [Object detection with Custom Vision](https://docs.microsoft.com/azure/cognitive-services/custom-vision-service/get-started-build-detector)
- [All Azure Cognitive Services](https://docs.microsoft.com/azure/cognitive-services/)
