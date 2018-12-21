# Spell Corrector PT
Correct automatically words in Portuguese.

# How to use
- Get word list (best to use domain-specific words to lower the computational costs)
- Train the Model (check out [example-train.py](https://github.com/samuelhei/spell-corrector-pt/blob/master/example-train.py))
- Specify the path to save the model to reuse afterward.
- Load the Model and correct the words (check out [example.py](https://github.com/samuelhei/spell-corrector-pt/blob/master/example.py))

# How the model works (high level)
- Preprocess the dictionary removing accentuation and transform to lowercase
- Extract char n_grams from the dictionary
- Create a sparse matrix from the dictionary utilizing the Bag of Words strategy
- Create a sparse matrix from the word preprocessed
- Compare the two sparse matrices by cosine similarity
- Return the most similar word
