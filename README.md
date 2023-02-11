# NLP-content-moderation
This is a content moderation system that uses NLP and deep learning techniques to classify a inputed text in different classes that are:


- Spam
  - [ ] Click-bait
  - [ ] Bot generated content
- Toxic 
  - [ ] Contains bad word
  - [ ] Offensive content
  - [ ] Explicit content
  - [ ] Violent content

To achieve this task I am going to create a NLP processing pipeline using Spacy library and I am then going to create a FastAPI.

## Development steps
- data acquisition
- preprocessing
- model creation
- api development

### Data acquisition
Acquiring representative and roubst dataset is always the most important point when it cames to create a machine learning model. In this case we will rely on free data from different source e.g. Kaggle.

In my case, since we have to train different classifiers, I am going to look for dataset that can provide me with at least 1000 observation for the following categories:

- spam
- bot generated text
- toxic text

### Preprocessing
Preproccessing is another key aspect, as a matter of fact, we are going to use the preprocessing part to filter some words and thus to classify bad content straight from here, without the need to pass the text in the ML models the could slow down the process.

### Model creation
Regarding the model 

### API development

## Pipeline structure

### Preprocessing

### Spam detection

### Toxicity detection

## API


## Citations
