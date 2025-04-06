import sys
from math import log2
from collections import defaultdict
from typing import List, Dict, Set, Tuple

class DocumentClassifier:
    
    def __init__(self):
        self.label_counts = defaultdict(int)
        self.token_counts = defaultdict(lambda: defaultdict(int))

        self.stop_words = {
            'about', 'all', 'along', 'also', 'although', 'among', 'and', 'any', 'anyone', 
            'anything', 'are', 'around', 'because', 'been', 'before', 'being', 'both', 
            'but', 'came', 'come', 'coming', 'could', 'did', 'each', 'else', 'every', 
            'for', 'from', 'get', 'getting', 'going', 'got', 'gotten', 'had', 'has', 
            'have', 'having', 'her', 'here', 'hers', 'him', 'his', 'how', 'however', 
            'into', 'its', 'like', 'may', 'most', 'next', 'now', 'only', 'our', 'out', 
            'particular', 'same', 'she', 'should', 'some', 'take', 'taken', 'taking', 
            'than', 'that', 'the', 'then', 'there', 'these', 'they', 'this', 'those', 
            'throughout', 'too', 'took', 'very', 'was', 'went', 'what', 'when', 'which', 
            'while', 'who', 'why', 'will', 'with', 'without', 'would', 'yes', 'yet', 
            'you', 'your',

            'com', 'doc', 'edu', 'encyclopedia', 'fact', 'facts', 'free', 'home', 'htm',
            'html', 'http', 'information', 'internet', 'net', 'new', 'news', 'official',
            'page', 'pages', 'resource', 'resources', 'pdf', 'site', 'sites', 'usa',
            'web', 'wikipedia', 'www',

            'one', 'ones', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
            'nine', 'ten', 'tens', 'eleven', 'twelve', 'dozen', 'dozens', 'thirteen',
            'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
            'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty',
            'ninety', 'hundred', 'hundreds', 'thousand', 'thousands', 'million',
            'millions'
        }

        self.stop_words.update(chr(i) for i in range(97, 123))
        
        self.stop_words.update(chr(i) + chr(j) for i in range(97, 123) for j in range(97, 123))

        self.labels = set()
        self.terms = set()

    def _clean_text(self, raw_text: str) -> List[str]:
        raw_text = raw_text.lower()
        raw_text = raw_text.replace(',', ' ').replace('.', ' ').replace('  ', ' ')

        tokens = raw_text.split()

        additional_stopwords = {'and', 'a', 'an', 'the', 'us', 'of', 'in', 'at', 'to'}
        self.stop_words.update(additional_stopwords)

        filtered_tokens = [token.strip() for token in tokens
                           if token not in self.stop_words
                           and len(token) > 1
                           and not token.isdigit()]

        return filtered_tokens

    def load_corpus(self, filename: str, train_size: int) -> Tuple[List[Dict], List[Dict]]:
        documents = []
        current_doc = {}
        current_tokens = []

        try:
            with open(filename, 'r') as file:
                lines = file.readlines()

            for line in lines:
                line = line.strip()

                if not line and current_tokens:
                    current_doc['text'] = ' '.join(token for token in current_tokens)
                    documents.append(current_doc)
                    current_doc = {}
                    current_tokens = []
                    continue

                if not line:
                    continue

                if 'name' not in current_doc:
                    current_doc['name'] = line
                elif 'label' not in current_doc:
                    current_doc['label'] = line
                    self.labels.add(line)
                else:
                    cleaned_tokens = self._clean_text(line)
                    current_tokens.extend(cleaned_tokens)

            if current_tokens:
                current_doc['text'] = ' '.join(current_tokens)
                documents.append(current_doc)

            training_set = documents[:train_size]
            self.terms = set()

            for doc in training_set:
                tokens = doc['text'].split()
                self.terms.update(tokens)

            return training_set, documents[train_size:]

        except FileNotFoundError:
            print(f"Error: Corpus file '{filename}' not found.")
            sys.exit(1)

    def compute_counts(self, training_set: List[Dict]):
        for doc in training_set:
            label = doc['label']
            self.label_counts[label] += 1
            unique_tokens = set(doc['text'].split())
            for token in unique_tokens:
                self.token_counts[label][token] += 1

    def calculate_probabilities(self, training_set: List[Dict]):
        e = 0.1 #epsilon
        num_labels = len(self.labels)

        total_docs = len(training_set)
        self.label_probs = {}
        for label in self.labels:
            frequency = self.label_counts[label] / total_docs
            self.label_probs[label] = (frequency + e ) / (1 + num_labels * e )

        self.token_probs = defaultdict(dict)
        for label in self.labels:
            label_count = self.label_counts[label]
            for token in self.terms:
                frequency = self.token_counts[label].get(token, 0) / label_count
                self.token_probs[label][token] = (frequency + e ) / (1 + 2 * e )

        self.log_label_probs = {
            label: -log2(prob) for label, prob in self.label_probs.items()
        }

        self.log_token_probs = defaultdict(dict)
        for label in self.labels:
            for token in self.terms:
                self.log_token_probs[label][token] = -log2(self.token_probs[label][token])

    def classify_documents(self, test_set: List[Dict]) -> List[Tuple[str, str, str, Dict[str, float]]]:
        results = []

        for doc in test_set:
            doc_name = doc['name']
            actual_label = doc['label']
            doc_text = doc['text']

            scores = {}
            tokens = set(doc_text.split())

            for label in self.labels:
                score = self.log_label_probs[label]

                for token in tokens:
                    if token in self.terms:
                        score += self.log_token_probs[label][token]

                scores[label] = score

            predicted_label = min(scores, key=scores.get)

            exponents = [2 ** (min(scores.values()) - score) for score in scores.values()]
            total_exp = sum(exponents)
            probabilities = {label: exp / total_exp for label, exp in zip(self.labels, exponents)}

            results.append((doc_name, actual_label, predicted_label, probabilities))

        return results

    def train(self, training_set: List[Dict]):
        self.compute_counts(training_set)
        self.calculate_probabilities(training_set)


def main():
    if len(sys.argv) != 3:
        print("Usage: python textClassifier.py <corpus_file> <train_size>")
        sys.exit(1)

    corpus_file = sys.argv[1]
    train_size = int(sys.argv[2])

    classifier = DocumentClassifier()
    train_set, test_set = classifier.load_corpus(corpus_file, train_size)

    classifier.train(train_set)

    predictions = classifier.classify_documents(test_set)

    correct_predictions = 0
    for doc_name, actual_label, predicted_label, probabilities in predictions:
        correctness = "Right" if actual_label == predicted_label else "Wrong"
        label_probs = " ".join([f"{label}: {prob:.2f}" for label, prob in probabilities.items()])
        print(f"{doc_name}. Prediction: {predicted_label}. {correctness}.")
        print(label_probs)
        if actual_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_set)
    print(f"\nOverall accuracy: {correct_predictions} out of {len(test_set)} = {accuracy:.2f}.")

if __name__ == "__main__":
    main()