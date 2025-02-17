import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

text = 'The man has a gun with him .'

doc = nlp(text)
print(text)

print("\n")
#tokenization
print('tokenization')
for token in doc:
    print(token.text)


print("\n")
#filter without stopwords
filtered_words = [token.text for token in doc if not token.is_stop]
print(filtered_words)


print("\n")
# Pos-Tags
print('Pos- Tags')
for tokens in doc:
    print(tokens.text, tokens.pos_)


print("\n")
#named entitty Recognition
print("Entity REcg")
for ent in doc.ents:
    print(ent.text, ent.label_)



print("\n")
#Lemmetization
print('lemmetization')
lemma = [tokens.lemma_ for tokens in doc if not tokens.is_punct]
print(lemma)
