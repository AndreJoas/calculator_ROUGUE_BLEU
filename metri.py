from flask import Flask, request, jsonify, render_template
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from collections import Counter
import nltk

nltk.download('punkt')  # Para tokenizar o texto
nltk.download('stopwords')  # Para as palavras de parada, se necessário
nltk.download('averaged_perceptron_tagger')  # Para marcação POS, se necessário

app = Flask(__name__)


with open("   NOME_DO_ARQUIVO.JSON  AQUI    ", "r", encoding="utf-8") as file:
    data = json.load(file)["examples"]

reference_answers = [item["reference_answer"] for item in data]

                                    # Função    " resposta mais semelhante"
def find_most_similar(input_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([input_text] + reference_answers)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    max_index = np.argmax(cosine_sim)
    return reference_answers[max_index]

    
    
                                            # Função ROUGE
def calculateRouge(text1, text2):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(text1, text2)
    
                             # precision, recall e F1
    rouge_metrics = {}
    for key, value in scores.items():
        rouge_metrics[key] = {
            "precision": value.precision,
            "recall": value.recall,
            "f1": value.fmeasure
        }
    return rouge_metrics


                                            #Função  BLUE 
def calculate_blue(reference, candidate):
                                 # Unigrams e Bigrams
    reference_unigrams = nltk.word_tokenize(reference.lower())
    candidate_unigrams = nltk.word_tokenize(candidate.lower())
    
    reference_bigrams = list(nltk.bigrams(reference_unigrams))
    candidate_bigrams = list(nltk.bigrams(candidate_unigrams))
    
                                         # BLUE-1: Precison com Unigrams
    precision_1 = len(set(candidate_unigrams) & set(reference_unigrams)) / len(set(candidate_unigrams)) if candidate_unigrams else 0
    print(precision_1)
                                    # BLUE-2: Precision com Bigrams
    precision_2 = len(set(candidate_bigrams) & set(reference_bigrams)) / len(set(candidate_bigrams)) if candidate_bigrams else 0
    
                                 # BLUE-L: A maior sobreposição de sequência de palavras
    common_sequence_len = 0
    for length in range(1, min(len(reference_unigrams), len(candidate_unigrams)) + 1):
        ref_subseq = set([tuple(reference_unigrams[i:i + length]) for i in range(len(reference_unigrams) - length + 1)])
        cand_subseq = set([tuple(candidate_unigrams[i:i + length]) for i in range(len(candidate_unigrams) - length + 1)])
        common_subseq = ref_subseq & cand_subseq
        common_sequence_len = max(common_sequence_len, len(common_subseq))
    
    precision_L = common_sequence_len / len(reference_unigrams) if len(reference_unigrams) else 0

    recall_1 = len(set(candidate_unigrams) & set(reference_unigrams)) / len(set(reference_unigrams)) if reference_unigrams else 0
    recall_2 = len(set(candidate_bigrams) & set(reference_bigrams)) / len(set(reference_bigrams)) if reference_bigrams else 0
    
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) != 0 else 0
    f1_2 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) != 0 else 0
    f1_L = 2 * (precision_L * (common_sequence_len / len(reference_unigrams))) / (precision_L + (common_sequence_len / len(reference_unigrams))) if (precision_L + (common_sequence_len / len(reference_unigrams))) != 0 else 0

    # métricas para BLUE-1, BLUE-2 e BLUE-L com Recall e F1
    blue_metrics = {
        "blue1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "blue2": precision_2,
        "recall_2": recall_2,
        "f1_2": f1_2,
        "blueL": precision_L,
        "recall_L": (common_sequence_len / len(reference_unigrams)),
        "f1_L": f1_L
    }
    return blue_metrics


@app.route("/")
def index():
    return render_template("TESTE.html")

@app.route("/find", methods=["POST"])
def find():
    input_text = request.json.get("input_text", "")
    most_similar = find_most_similar(input_text)
    return jsonify({"similar_answer": most_similar})

@app.route("/rouge", methods=["POST"])
def rouge():
    text1 = request.json.get("text1", "")
    text2 = request.json.get("text2", "")
    rouge_scores = calculateRouge(text1, text2)
    return jsonify(rouge_scores)

@app.route("/blue", methods=["POST"])
def blue():
    reference_text = request.json.get("reference_text", "")
    candidate_text = request.json.get("candidate_text", "")
    blue_scores = calculate_blue(reference_text, candidate_text)
    return jsonify(blue_scores)

if __name__ == "__main__":
    app.run(debug=True)
