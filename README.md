# arXiv Benchmark Paper Analyzer

A comprehensive tool for automatically fetching, classifying, and analyzing benchmark papers from arXiv.

## Features

- **Automatic Paper Fetching**: Fetch papers from arXiv based on keywords and date range
- **Local Model Classification**: Use locally deployed transformers models to determine if papers are benchmarks
- **Large Model Analysis**: Detailed analysis of benchmark papers using LLM APIs
- **Multi-label Classification**: Classify papers with 50+ predefined labels covering various domains
- **Database Storage**: SQLite database for persistent storage of papers and analysis results
- **Interactive Visualization**: Streamlit-based dashboard for exploring results and label intersections

## Classification Labels

The system supports the following classification categories:

- **知识类**: 综合知识, 前沿与边缘知识
- **能力类**: 逻辑与推理, 数学, 代码, 软件工程
- **行业类**: 法律, 金融, 医疗, 科学发现, 生物与基因, 电信, 人力资源, 教育, 供应链与物流, 文化
- **Agent能力**: 工具使用, 环境交互与规划, 多智能体协作, 游戏智能
- **RAG与长文本**: RAG系统鲁棒性, 长文本理解, 学术与论文分析
- **多模态**: 视觉-语言, 文字文本, 视频理解与生成, 音频与语音, 3D与具身智能
- **安全与伦理**: 机器遗忘, 版权与合规, 安全与越狱, 幻觉, 欺骗与奉承
- **心理与认知**: 情商与人格, 认知偏差
- **可解释与鲁棒性**: 可解释性与忠实度, 提示词鲁棒性
- **性能与效率**: 推理性能, 绿色AI (能效), 水印鲁棒性
- **特殊数据类型**: 表格理解, 时间序列, 图结构, 推荐系统
- **评估方法**: LLM裁判, 数据污染检测, 不确定性校准

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.py` or set environment variables:

```bash
# For large model API (optional)
export OPENAI_API_KEY="your-api-key"
export API_BASE="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo-16k"

# For local model (default: facebook/bart-large-mnli)
export LOCAL_CLASSIFIER_MODEL="facebook/bart-large-mnli"
```

## Usage

### 1. Run the Analyzer

```bash
python arxiv_analyzer.py
```

This will:
1. Fetch papers from arXiv
2. Classify papers as benchmark/non-benchmark using the local model
3. Download PDFs for benchmark papers
4. Analyze with large model API (if configured)
5. Store results in SQLite database
6. Export results to markdown and JSON

### 2. Initialize Database (Optional)

```bash
python database.py
```

### 3. Launch Visualization Dashboard

```bash
streamlit run visualization.py
```

This opens a web interface at `http://localhost:8501` with:
- **Statistics Dashboard**: Overview of papers and label distribution
- **Label Matrix**: Interactive intersection matrix for selected labels
- **Browse Papers**: Search and filter papers by labels

## Project Structure

```
├── config.py              # Configuration settings and label definitions
├── database.py            # SQLAlchemy ORM models and database utilities
├── local_classifier.py    # Local model classifier using transformers
├── arxiv_analyzer.py      # Main analysis pipeline
├── visualization.py       # Streamlit visualization dashboard
├── requirements.txt       # Python dependencies
├── arxiv_pdfs/           # Downloaded PDF files
├── output/               # Exported analysis results
└── arxiv_benchmark.db    # SQLite database
```

## API Reference

### ArxivBenchmarkAnalyzer

```python
from arxiv_analyzer import ArxivBenchmarkAnalyzer

# Initialize
analyzer = ArxivBenchmarkAnalyzer(use_local_model=True)

# Fetch papers
papers = analyzer.fetch_arxiv_papers(
    keyword="benchmark",
    days=7,
    max_results=100
)

# Process papers
stats = analyzer.process_papers(papers)

# Get label statistics
label_stats = analyzer.get_label_statistics()

# Get intersection matrix
matrix = analyzer.get_label_intersection_matrix(['Label1', 'Label2'])

# Export results
analyzer.export_results('markdown')
analyzer.export_results('json')

# Close
analyzer.close()
```

### LocalBenchmarkClassifier

```python
from local_classifier import LocalBenchmarkClassifier

classifier = LocalBenchmarkClassifier(model_name="facebook/bart-large-mnli")

is_benchmark, confidence, details = classifier.is_benchmark_paper(
    title="Paper Title",
    abstract="Paper abstract...",
    threshold=0.5
)
```

## Label Intersection Matrix

The visualization dashboard provides an interactive matrix showing the intersection of papers across different labels:

- Select multiple labels using checkboxes
- View the heatmap showing how many papers have both labels
- Download the matrix as CSV for further analysis

## Notes

- Local model inference requires sufficient GPU memory for optimal performance
- First run may take time to download model weights
- API calls to large models may incur costs
- Default benchmark classification threshold is 0.5

## License

MIT License