"""
Local model classifier for benchmark paper detection
Uses transformers for local inference
"""
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


class LocalBenchmarkClassifier:
    """
    Local model classifier using transformers to determine if a paper is a benchmark paper.
    Uses zero-shot classification approach.
    """
    
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        """
        Initialize the local classifier.
        
        Args:
            model_name: HuggingFace model name for zero-shot classification
            device: Device to run inference on (None for auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        
        print(f"Initializing local classifier with model: {model_name}")
        print(f"Using device: {'GPU' if self.device >= 0 else 'CPU'}")
        
        # Initialize zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device
        )
        
        # Candidate labels for benchmark detection
        self.benchmark_labels = [
            "benchmark dataset",
            "evaluation benchmark",
            "performance evaluation",
            "model comparison",
            "not a benchmark"
        ]
        
        print("Local classifier initialized successfully!")
    
    def is_benchmark_paper(self, title, abstract, threshold=0.5):
        """
        Determine if a paper is a benchmark paper based on title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract/summary
            threshold: Confidence threshold for classification
            
        Returns:
            tuple: (is_benchmark: bool, confidence: float, details: dict)
        """
        # Combine title and abstract for classification
        text = f"Title: {title}\n\nAbstract: {abstract}"
        
        # Truncate if too long (model has max length limit)
        max_length = 1024
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            # Run classification
            result = self.classifier(
                text,
                candidate_labels=self.benchmark_labels,
                multi_label=True
            )
            
            # Check if any benchmark-related label has high confidence
            benchmark_positive_labels = [
                "benchmark dataset",
                "evaluation benchmark",
                "performance evaluation",
                "model comparison"
            ]
            
            # Sum up positive benchmark scores
            benchmark_score = 0
            for label, score in zip(result['labels'], result['scores']):
                if label in benchmark_positive_labels:
                    benchmark_score += score
            
            # Normalize score
            benchmark_score = min(benchmark_score, 1.0)
            
            is_benchmark = benchmark_score >= threshold
            
            details = {
                'labels': result['labels'],
                'scores': result['scores'],
                'benchmark_score': benchmark_score
            }
            
            return is_benchmark, benchmark_score, details
            
        except Exception as e:
            print(f"Classification error: {e}")
            return False, 0.0, {'error': str(e)}


class LocalLabelClassifier:
    """
    Local model classifier for multi-label classification of benchmark papers.
    """
    
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        """
        Initialize the label classifier.
        
        Args:
            model_name: HuggingFace model name for zero-shot classification
            device: Device to run inference on
        """
        self.model_name = model_name
        
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        
        print(f"Initializing label classifier with model: {model_name}")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device
        )
        
        print("Label classifier initialized successfully!")
    
    def classify_paper(self, title, abstract, candidate_labels, threshold=0.3, top_k=5):
        """
        Classify a paper with multiple labels.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            candidate_labels: List of candidate labels
            threshold: Minimum confidence threshold
            top_k: Maximum number of labels to return
            
        Returns:
            list: List of (label, score) tuples for matched labels
        """
        text = f"Title: {title}\n\nAbstract: {abstract}"
        
        # Truncate if too long
        max_length = 1024
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            result = self.classifier(
                text,
                candidate_labels=candidate_labels,
                multi_label=True
            )
            
            # Filter by threshold and take top k
            matched_labels = []
            for label, score in zip(result['labels'], result['scores']):
                if score >= threshold:
                    matched_labels.append((label, score))
            
            # Sort by score and take top k
            matched_labels.sort(key=lambda x: x[1], reverse=True)
            return matched_labels[:top_k]
            
        except Exception as e:
            print(f"Label classification error: {e}")
            return []


def test_classifier():
    """Test function for the classifier"""
    # Sample benchmark paper
    title = "MMLU: A Massive Multitask Language Understanding Benchmark"
    abstract = """We introduce a new test to measure a text model's multitask accuracy. 
    The test covers 57 tasks including elementary mathematics, US history, computer science, 
    law, and more. We show that recent models still have significant room for improvement."""
    
    # Sample non-benchmark paper
    title2 = "Attention Is All You Need"
    abstract2 = """We propose a new simple network architecture, the Transformer, 
    based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."""
    
    print("Testing LocalBenchmarkClassifier...")
    classifier = LocalBenchmarkClassifier()
    
    print("\n--- Test 1: Benchmark paper ---")
    is_bench, conf, details = classifier.is_benchmark_paper(title, abstract)
    print(f"Is benchmark: {is_bench}, Confidence: {conf:.3f}")
    
    print("\n--- Test 2: Non-benchmark paper ---")
    is_bench, conf, details = classifier.is_benchmark_paper(title2, abstract2)
    print(f"Is benchmark: {is_bench}, Confidence: {conf:.3f}")


if __name__ == "__main__":
    test_classifier()
