"""
arXiv Benchmark Paper Analyzer

This script:
1. Fetches papers from arXiv
2. Uses a local model to classify if papers are benchmarks
3. Downloads and analyzes benchmark papers with a large model
4. Classifies papers with predefined labels
5. Stores results in SQLite database
"""

import arxiv
import requests
from datetime import datetime, timedelta
import time
import json
import os
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
import PyPDF2

from config import (
    API_KEY, API_BASE, MODEL_NAME,
    LOCAL_CLASSIFIER_MODEL,
    PDF_DIR, DATABASE_PATH, OUTPUT_DIR,
    DEFAULT_SEARCH_DAYS, DEFAULT_MAX_RESULTS,
    CLASSIFICATION_LABELS, LABEL_CATEGORIES
)
from database import (
    get_session, init_database, init_labels,
    Paper, Label, AnalysisLog
)
from local_classifier import LocalBenchmarkClassifier, LocalLabelClassifier


class ArxivBenchmarkAnalyzer:
    """Main class for arXiv benchmark paper analysis"""
    
    def __init__(self, use_local_model=True, local_model_name=None):
        """
        Initialize the analyzer.
        
        Args:
            use_local_model: Whether to use local model for benchmark classification
            local_model_name: Name of the local model to use
        """
        print("="*60)
        print("arXiv Benchmark Paper Analyzer")
        print("="*60)
        
        # Initialize database
        init_database()
        self.session = get_session()
        init_labels(self.session, CLASSIFICATION_LABELS, LABEL_CATEGORIES)
        
        # Initialize local classifier if enabled
        self.use_local_model = use_local_model
        self.benchmark_classifier = None
        self.label_classifier = None
        
        if use_local_model:
            model_name = local_model_name or LOCAL_CLASSIFIER_MODEL
            print(f"\nInitializing local models...")
            try:
                self.benchmark_classifier = LocalBenchmarkClassifier(model_name)
                self.label_classifier = LocalLabelClassifier(model_name)
            except Exception as e:
                print(f"Warning: Could not initialize local models: {e}")
                print("Falling back to API-based classification")
                self.use_local_model = False
        
        print("\nAnalyzer initialized successfully!")
    
    def fetch_arxiv_papers(self, keyword="benchmark", days=None, max_results=None, 
                           categories=None):
        """
        Fetch papers from arXiv.
        
        Args:
            keyword: Search keyword
            days: Number of days to search back
            max_results: Maximum number of results
            categories: List of arXiv categories to filter (e.g., ['cs.AI', 'cs.CL'])
            
        Returns:
            list: List of paper dictionaries
        """
        days = days or DEFAULT_SEARCH_DAYS
        max_results = max_results or DEFAULT_MAX_RESULTS
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Build query
        query_parts = []
        if keyword:
            query_parts.append(f'all:{keyword}')
        
        if categories:
            cat_query = ' OR '.join([f'cat:{cat}' for cat in categories])
            query_parts.append(f'({cat_query})')
        
        query = ' AND '.join(query_parts) if query_parts else 'all:*'
        
        print(f"\nSearching arXiv papers...")
        print(f"Query: {query}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Max results: {max_results}")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        try:
            for result in tqdm(client.results(search), desc="Fetching papers"):
                paper = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'published': result.published.strftime("%Y-%m-%d"),
                    'updated': result.updated.strftime("%Y-%m-%d"),
                    'url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'categories': result.categories,
                    'arxiv_id': result.get_short_id()
                }
                papers.append(paper)
                
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        print(f"Found {len(papers)} papers")
        return papers
    
    def classify_benchmark(self, paper, threshold=0.5):
        """
        Classify if a paper is a benchmark paper.
        
        Args:
            paper: Paper dictionary
            threshold: Confidence threshold
            
        Returns:
            tuple: (is_benchmark, confidence, details)
        """
        if self.use_local_model and self.benchmark_classifier:
            # Use local model
            return self.benchmark_classifier.is_benchmark_paper(
                paper['title'],
                paper['summary'],
                threshold=threshold
            )
        else:
            # Fallback: simple keyword-based classification
            benchmark_keywords = [
                'benchmark', 'evaluation', 'dataset', 'test suite',
                'leaderboard', 'comparison', 'assessment'
            ]
            
            text = (paper['title'] + ' ' + paper['summary']).lower()
            matches = sum(1 for kw in benchmark_keywords if kw in text)
            confidence = min(matches / 3, 1.0)
            is_benchmark = confidence >= threshold
            
            return is_benchmark, confidence, {'method': 'keyword', 'matches': matches}
    
    def classify_labels(self, paper, paper_text=None, threshold=0.3, top_k=5):
        """
        Classify a paper with predefined labels.
        
        Args:
            paper: Paper dictionary
            paper_text: Full paper text (optional)
            threshold: Confidence threshold
            top_k: Maximum number of labels
            
        Returns:
            list: List of (label, score) tuples
        """
        if self.use_local_model and self.label_classifier:
            return self.label_classifier.classify_paper(
                paper['title'],
                paper['summary'],
                CLASSIFICATION_LABELS,
                threshold=threshold,
                top_k=top_k
            )
        else:
            # Fallback: return empty list
            return []
    
    def download_pdf(self, pdf_url, save_path):
        """
        Download PDF file.
        
        Args:
            pdf_url: PDF download URL
            save_path: Path to save the file
            
        Returns:
            bool: Whether download succeeded
        """
        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"PDF download failed: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_path, max_pages=20):
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to read
            
        Returns:
            str: Extracted text or None if failed
        """
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
                pages_to_read = min(max_pages, total_pages) if max_pages else total_pages
                
                text = ""
                for page_num in range(pages_to_read):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text
        except Exception as e:
            print(f"PDF text extraction failed: {e}")
            return None
    
    def truncate_text(self, text, max_tokens=12000):
        """Truncate text to fit model token limit"""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[Content truncated due to length limit]"
    
    def analyze_with_large_model(self, paper, paper_text):
        """
        Analyze paper with large model API.
        
        Args:
            paper: Paper dictionary
            paper_text: Full paper text
            
        Returns:
            str: Analysis result
        """
        if API_KEY == "your-api-key-here":
            return "API key not configured. Skipping large model analysis."
        
        truncated_text = self.truncate_text(paper_text)
        
        # Create label list string
        labels_str = "\n".join([f"- {label}" for label in CLASSIFICATION_LABELS])
        
        prompt = f"""
Please analyze the following academic paper in detail:

Title: {paper['title']}

Full Paper Text:
{truncated_text}

Please analyze according to the following structure:

1. **Research Area**: Main research domain

2. **Research Problem**: What problem does this paper address?

3. **Main Method**: What method or model does the paper propose?

4. **Benchmark Information**:
   - What benchmark datasets are used?
   - Is a new benchmark proposed?
   - What are the performance results?

5. **Main Contributions**: (3-5 points)

6. **Experimental Results**: Key performance metrics and conclusions

7. **Innovation**: Novel aspects compared to existing work

8. **Limitations**: Shortcomings or future improvements

9. **Practical Value**: Application prospects and value

10. **Classification Labels**: From the following list, select ALL applicable labels:
{labels_str}

Please output in the format:
LABELS: [label1], [label2], [label3], ...

11. **Overall Evaluation**: (~200 words comprehensive evaluation)

Please provide detailed and professional analysis.
"""
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior AI researcher and paper reviewer, skilled at in-depth analysis of academic papers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2500
        }
        
        try:
            response = requests.post(
                f"{API_BASE}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def save_paper_to_db(self, paper, is_benchmark, confidence, labels, analysis=None):
        """
        Save paper information to database.
        
        Args:
            paper: Paper dictionary
            is_benchmark: Whether it's a benchmark paper
            confidence: Benchmark classification confidence
            labels: List of label names
            analysis: Full analysis text
            
        Returns:
            Paper: Saved paper object
        """
        # Check if paper already exists
        existing = self.session.query(Paper).filter_by(arxiv_id=paper['arxiv_id']).first()
        
        if existing:
            # Update existing paper
            db_paper = existing
            db_paper.is_benchmark = is_benchmark
            db_paper.benchmark_confidence = str(confidence)
            db_paper.full_analysis = analysis
            db_paper.updated_at = datetime.utcnow()
        else:
            # Create new paper
            db_paper = Paper(
                arxiv_id=paper['arxiv_id'],
                title=paper['title'],
                authors=json.dumps(paper['authors']),
                summary=paper['summary'],
                published=paper['published'],
                updated=paper['updated'],
                url=paper['url'],
                pdf_url=paper['pdf_url'],
                categories=json.dumps(paper['categories']),
                is_benchmark=is_benchmark,
                benchmark_confidence=str(confidence),
                full_analysis=analysis
            )
            self.session.add(db_paper)
        
        # Add labels
        if labels:
            db_paper.labels = []  # Clear existing labels
            for label_name in labels:
                if isinstance(label_name, tuple):
                    label_name = label_name[0]  # Handle (label, score) tuples
                
                label = self.session.query(Label).filter_by(name=label_name).first()
                if label:
                    db_paper.labels.append(label)
        
        self.session.commit()
        return db_paper
    
    def process_papers(self, papers, download_pdf=True, analyze_with_llm=True,
                      benchmark_threshold=0.5):
        """
        Process a list of papers through the full pipeline.
        
        Args:
            papers: List of paper dictionaries
            download_pdf: Whether to download PDFs for benchmark papers
            analyze_with_llm: Whether to analyze with large model
            benchmark_threshold: Threshold for benchmark classification
            
        Returns:
            dict: Processing statistics
        """
        stats = {
            'total': len(papers),
            'benchmark': 0,
            'analyzed': 0,
            'failed': 0
        }
        
        print(f"\nProcessing {len(papers)} papers...")
        
        for i, paper in enumerate(tqdm(papers, desc="Processing papers")):
            print(f"\n{'='*60}")
            print(f"Paper {i+1}/{len(papers)}: {paper['title'][:60]}...")
            
            # Step 1: Classify if benchmark
            is_benchmark, confidence, details = self.classify_benchmark(
                paper, threshold=benchmark_threshold
            )
            
            print(f"  Benchmark: {is_benchmark} (confidence: {confidence:.3f})")
            
            labels = []
            analysis = None
            
            if is_benchmark:
                stats['benchmark'] += 1
                
                # Step 2: Get labels using local model
                if self.use_local_model:
                    labels = self.classify_labels(paper)
                    print(f"  Labels: {[l[0] for l in labels]}")
                
                # Step 3: Download PDF and analyze (if enabled)
                if download_pdf and analyze_with_llm:
                    pdf_filename = f"{paper['arxiv_id'].replace('/', '_')}.pdf"
                    pdf_path = PDF_DIR / pdf_filename
                    
                    if self.download_pdf(paper['pdf_url'], pdf_path):
                        paper_text = self.extract_text_from_pdf(pdf_path)
                        
                        if paper_text:
                            print(f"  Analyzing with large model...")
                            analysis = self.analyze_with_large_model(paper, paper_text)
                            stats['analyzed'] += 1
                            
                            # Extract labels from analysis if available
                            if "LABELS:" in analysis:
                                # Parse labels from analysis
                                pass
                        else:
                            stats['failed'] += 1
                    else:
                        stats['failed'] += 1
            
            # Save to database
            label_names = [l[0] if isinstance(l, tuple) else l for l in labels]
            self.save_paper_to_db(paper, is_benchmark, confidence, label_names, analysis)
            
            # Rate limiting
            time.sleep(1)
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"  Total papers: {stats['total']}")
        print(f"  Benchmark papers: {stats['benchmark']}")
        print(f"  Successfully analyzed: {stats['analyzed']}")
        print(f"  Failed: {stats['failed']}")
        
        return stats
    
    def export_results(self, output_format='markdown'):
        """
        Export analysis results.
        
        Args:
            output_format: 'markdown', 'json', or 'csv'
            
        Returns:
            str: Path to output file
        """
        papers = self.session.query(Paper).filter_by(is_benchmark=True).all()
        
        if output_format == 'markdown':
            output_file = OUTPUT_DIR / "arxiv_benchmark_papers.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# arXiv Benchmark Papers Analysis\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Total benchmark papers: {len(papers)}\n\n")
                f.write("---\n\n")
                
                for i, paper in enumerate(papers, 1):
                    f.write(f"## {i}. {paper.title}\n\n")
                    f.write(f"**arXiv ID:** {paper.arxiv_id}\n\n")
                    f.write(f"**Published:** {paper.published}\n\n")
                    f.write(f"**Authors:** {', '.join(json.loads(paper.authors)[:5])}\n\n")
                    f.write(f"**URL:** [{paper.url}]({paper.url})\n\n")
                    f.write(f"**Categories:** {', '.join(json.loads(paper.categories))}\n\n")
                    f.write(f"**Benchmark Confidence:** {paper.benchmark_confidence}\n\n")
                    
                    if paper.labels:
                        f.write(f"**Labels:** {', '.join([l.name for l in paper.labels])}\n\n")
                    
                    if paper.full_analysis:
                        f.write(f"### Analysis\n\n{paper.full_analysis}\n\n")
                    
                    f.write("---\n\n")
            
            print(f"Results exported to: {output_file}")
            return str(output_file)
        
        elif output_format == 'json':
            output_file = OUTPUT_DIR / "arxiv_benchmark_papers.json"
            data = []
            for paper in papers:
                data.append({
                    'arxiv_id': paper.arxiv_id,
                    'title': paper.title,
                    'authors': json.loads(paper.authors),
                    'summary': paper.summary,
                    'published': paper.published,
                    'url': paper.url,
                    'categories': json.loads(paper.categories),
                    'benchmark_confidence': paper.benchmark_confidence,
                    'labels': [l.name for l in paper.labels],
                    'analysis': paper.full_analysis
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Results exported to: {output_file}")
            return str(output_file)
        
        return None
    
    def get_label_statistics(self):
        """
        Get statistics for all labels.
        
        Returns:
            dict: Label name -> count mapping
        """
        labels = self.session.query(Label).all()
        stats = {}
        for label in labels:
            stats[label.name] = len(label.papers)
        return stats
    
    def get_label_intersection_matrix(self, selected_labels=None):
        """
        Get intersection matrix for selected labels.
        
        Args:
            selected_labels: List of label names (None for all)
            
        Returns:
            dict: Matrix data with labels and counts
        """
        if selected_labels is None:
            labels = self.session.query(Label).all()
        else:
            labels = self.session.query(Label).filter(
                Label.name.in_(selected_labels)
            ).all()
        
        label_names = [l.name for l in labels]
        matrix = {}
        
        for label1 in labels:
            matrix[label1.name] = {}
            papers1 = set(p.id for p in label1.papers)
            
            for label2 in labels:
                papers2 = set(p.id for p in label2.papers)
                intersection = len(papers1 & papers2)
                matrix[label1.name][label2.name] = intersection
        
        return {
            'labels': label_names,
            'matrix': matrix
        }
    
    def close(self):
        """Close database session"""
        self.session.close()


def main():
    """Main function"""
    print("="*60)
    print("arXiv Benchmark Paper Analyzer - Full Pipeline")
    print("="*60 + "\n")
    
    # Check API key
    if API_KEY == "your-api-key-here":
        print("⚠️  Warning: API key not configured.")
        print("Large model analysis will be skipped.")
        print("Set OPENAI_API_KEY environment variable or update config.py\n")
    
    # Initialize analyzer
    analyzer = ArxivBenchmarkAnalyzer(use_local_model=True)
    
    try:
        # Fetch papers
        papers = analyzer.fetch_arxiv_papers(
            keyword="benchmark",
            days=7,
            max_results=10  # Start with small number for testing
        )
        
        if not papers:
            print("No papers found")
            return
        
        # Process papers
        stats = analyzer.process_papers(
            papers,
            download_pdf=True,
            analyze_with_llm=True,
            benchmark_threshold=0.4
        )
        
        # Export results
        analyzer.export_results('markdown')
        analyzer.export_results('json')
        
        # Show label statistics
        print("\nLabel Statistics:")
        label_stats = analyzer.get_label_statistics()
        for label, count in sorted(label_stats.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {label}: {count}")
        
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
