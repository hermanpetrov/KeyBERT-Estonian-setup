import os
import re
import csv
from torch import cuda
from sentence_transformers import SentenceTransformer
from flair.embeddings import TransformerDocumentEmbeddings
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
import stanza




nlp = stanza.Pipeline(lang="et", processors="tokenize,lemma")

input_directory = "raw_text/"
output_directory = "raw_text_lemma/"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def clean_text(text):

    text = re.sub(r"[_=+]", "", text)
    return text


for entry in os.scandir(input_directory):
    if entry.is_file() and entry.name.endswith(".txt"):
        print("Processing file:", entry.name)

        output_file_path = os.path.join(output_directory, entry.name)

        with open(entry.path, "r", encoding="utf-8") as input_file:
            text = input_file.read()

        cleaned_text = clean_text(text)

        doc = nlp(cleaned_text)
        lemmatized_text = " ".join(
            [word.lemma for sent in doc.sentences for word in sent.words]
        )

        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(lemmatized_text)

        print("Finished processing file:", entry.name)


def load_and_preprocess_stopwords(file_path):
    with open(file_path, "r", encoding="UTF-8") as file:
        stopwords = [re.sub(r"\W+", "", line.strip().lower()) for line in file]
    return stopwords


def custom_tokenizer(doc):
    return re.split(r"[\s,.!?;:()]+", doc)


def read_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(folder_path, filename), "r", encoding="utf-8"
            ) as file:
                docs.append((filename, file.read()))
    return docs


def load_model(model_info):
    model_type, model_path = model_info
    if model_type == "sentence_transformer":
        model = SentenceTransformer(model_path)
    elif model_type == "flair_transformer":
        model = TransformerDocumentEmbeddings(model_path, device="cuda")
    return KeyBERT(model=model)


def run_models(
    docs,
    model,
    model_name,
    output_base,
    ngram_ranges,
    diversities,
    lowercase,
    batch_size=5,
):
    stopwords = load_and_preprocess_stopwords("estonian-stopwords.txt")
    for ngram_range in ngram_ranges:
        vectorizer = CountVectorizer(
            tokenizer=custom_tokenizer,
            ngram_range=ngram_range,
            stop_words=stopwords,
            token_pattern=None,
            lowercase=lowercase,
        )
        for diversity in diversities:
            output_dir_path = os.path.join(
                output_base,
                f"{model_name}",
                f"ngram_{ngram_range[0]}_{ngram_range[1]}",
                f"diversity_{int(diversity*10)}",
            )
            os.makedirs(output_dir_path, exist_ok=True)

            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i : i + batch_size]
                batch_texts = [text for _, text in batch_docs]
                batch_filenames = [filename for filename, _ in batch_docs]
                keywords_batch = [
                    model.extract_keywords(
                        doc,
                        use_mmr=True,
                        diversity=diversity,
                        vectorizer=vectorizer,
                        nr_candidates=200,
                        top_n=200,
                    )
                    for doc in batch_texts
                ]

                for keywords, filename in zip(keywords_batch, batch_filenames):
                    output_path = os.path.join(output_dir_path, f"{filename[:-4]}.csv")
                    with open(
                        output_path, "w", newline="", encoding="utf-8"
                    ) as csvfile:
                        writer = csv.writer(csvfile, delimiter=";")
                        writer.writerow(["keyphrase", "score"])
                        for keyphrase, score in keywords:
                            writer.writerow([keyphrase, score])

            print(
                f"Finished processing {model_name} at ngram range {ngram_range} and diversity {diversity} with nr_candidates=200 and top_n=200 and lowercase={lowercase}"
            )
    del model
    if cuda.is_available():
        cuda.empty_cache()


def main():
    base_folders = {
        "raw_text": "models/raw_text_data",
        "raw_text_lemma": "models/raw_text_lemma_data",
    }
    lcf_folders = {
        "raw_text": "models/raw_text_data_LCF",
        "raw_text_lemma": "models/raw_text_lemma_data_LCF",
    }
    models_info = {
        "LaBSE": ("sentence_transformer", "sentence-transformers/LaBSE"),
        "multi_e5": ("sentence_transformer", "intfloat/multilingual-e5-large-instruct"),
        "MiniLM-L12_multi": (
            "sentence_transformer",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    }
    ngram_ranges = [(1, 1)]
    diversities = [0.2]

    for folder_key in base_folders:
        folder_path = "raw_text" if "lemma" not in folder_key else "raw_text_lemma"
        docs = read_documents(folder_path)
        for model_name, model_info in models_info.items():
            model = load_model(model_info)
            run_models(
                docs,
                model,
                model_name,
                base_folders[folder_key],
                ngram_ranges,
                diversities,
                lowercase=True,
            )
            run_models(
                docs,
                model,
                model_name,
                lcf_folders[folder_key],
                ngram_ranges,
                diversities,
                lowercase=False,
            )


if __name__ == "__main__":
    main()
