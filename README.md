# Tyche: Probabilistic Selection over Encrypted Data for Generative Language Models

Tyche is a first look into fully homomorphic generative AI. Cloud-based generative language models are ridden with concerns over privacy, especially since details about the training data and process have been made proprietary. Many users and companies are scared their secret information is being used to train the model and may be leaked by others. 

Tyche is working to prevent that by using fully homomorphic encryption by allowing computations to be encrypted. Thus all inputs, outputs and computations along the way are encrypted by the user such that the AI cloud cannot leak information about them. This work evaluates several algorithms to perform probabilistic selection of outputs in generative AI models. Here, a model generates an encrypted prediction; say the next word "store" has a high probability and "elephant" has a low probability, but these probabilities are encrypted. Tyche explores algorithms to select the next *encrypted* word with the corresponding frequency.


