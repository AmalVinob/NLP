import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

sentences = [
    "Apple is looking at buying U.K. startup for $1 billion.",
    "Elon Musk announced that SpaceX will launch its next mission from Florida.",
    "The quick brown fox jumps over the lazy dog.",
    "Can you believe it’s already February 2025?",
    "Barack Obama was the 44th President of the United States.",
    "I ordered pizza from Domino's, but they delivered sushi instead.",
    "Python and spaCy are powerful tools for natural language processing.",
    "Despite the rain, she decided to go for a long walk in Central Park.",
    "The movie ‘Inception’ was directed by Christopher Nolan.",
    "He earned a Ph.D. in computer science from MIT last year."
]

token_list = []
filtered_token = []
pos_tag_list = []
ner_list = []


for sentence in sentences:
    doc = nlp(sentence.strip())


    # abse tokens without any stop word removal
    tokens = [words.text for words in doc]
    token_list.append(tokens)

    #tokenization after stopword removal
    fil_tokens = [word for word in doc if not word.is_stop]
    filtered_token.append(fil_tokens)

    #pos_taglist
    pos_tag = [(token.text, token.pos_) for token in doc]
    pos_tag_list.append(pos_tag)

    #ner
    ner = [(token.text, token.label_) for token in doc.ents]
    ner_list.append(ner)


result_df = pd.DataFrame({
    'sentiment_sentence' : sentences,
    'tokens' : token_list,
    'filterd_tokes' : filtered_token,
    'pos_tags' : pos_tag_list,
    'NER' : ner_list
})


print(result_df)

result_df.to_csv('processed_data.csv', index=False)

df = pd.read_csv('processed_data.csv', encoding='latin-1')
print(df.head())

