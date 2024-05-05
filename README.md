## Soccer Event Detection using CNN and LLava Language Model

### Overview:
This project aims to detect soccer events in images using Convolutional Neural Networks (CNN) and generate textual descriptions using LLava, an open-source language model. The dataset comprises 7000 images scraped from various sources, including web scraping and Kaggle. The project includes steps such as data gathering, exploratory data analysis (EDA), preprocessing, CNN model fitting, hyperparameter tuning, early stopping, and LLava LLM integration for image description generation and recommendation.

### Features:
- **Data Gathering:** The dataset consists of 7000 images collected from web scraping and Kaggle, categorized into seven soccer event classes.
- **CNN Model Fitting:** A CNN model is trained to classify images into different soccer event categories, achieving an overall accuracy of 74.59%.
- **Hyperparameter Tuning:** Hyperparameter tuning is performed using Keras Tuner to optimize the CNN model's performance.
- **LLava Language Model:** LLava, an open-source language model, is employed to generate textual descriptions for the soccer event images.
- **Image Recommendation:** Based on user interests, the LLava model recommends relevant soccer event images to the user.
- **Research Paper:** The project includes a research paper discussing the methodology, results, and further analysis.

### Steps:
1. **Data Collection:** Images are gathered from web scraping and Kaggle datasets, totaling 7000 images.
2. **EDA and Preprocessing:** Exploratory data analysis is performed to understand the dataset's characteristics, followed by preprocessing steps such as resizing and normalization.
3. **CNN Model Fitting:** A CNN model is trained on the preprocessed images to classify them into seven soccer event categories.
4. **Hyperparameter Tuning:** Keras Tuner is used to optimize the CNN model's hyperparameters for improved performance.
5. **Early Stopping:** Early stopping is applied during model training to prevent overfitting and improve generalization.
6. **LLava Integration:** LLava LLM is integrated to generate textual descriptions for the soccer event images.
7. **Image Recommendation:** Based on user interests, LLava recommends relevant soccer event images from the dataset.

### Further Analysis:
- The project includes a detailed analysis of model performance, including precision, recall, and F1-score metrics for each soccer event class.
- The LLava-generated descriptions are evaluated for coherence and relevance to the corresponding images.
- Future work may involve refining the CNN model architecture, exploring transfer learning techniques, and enhancing LLava's image description capabilities.

### Data Availability:
The dataset used in this project is available for download from my Kaggle account at the following link: https://www.kaggle.com/datasets/rishintiw/soccer-event-classification-image-data-cnn-and-llm

### Research Paper:
A detailed research paper discussing the project methodology, results, and insights is included in the project repository. link : https://github.com/RishinTiwari/Soccer-Event-Detection-using-CNN-and-Llava-LLM/blob/main/Final%20Report%20Group%204%20ISM%206215.pdf

### Repository Link:
The project code, datasets, research paper, and documentation are available on GitHub at : https://github.com/RishinTiwari/Soccer-Event-Detection-using-CNN-and-Llava-LLM
