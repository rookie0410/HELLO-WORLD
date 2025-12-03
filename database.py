"""
Database models for the arXiv Benchmark Analyzer
Uses SQLAlchemy for ORM
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Table, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from config import DATABASE_PATH

Base = declarative_base()

# Association table for many-to-many relationship between papers and labels
paper_labels = Table(
    'paper_labels',
    Base.metadata,
    Column('paper_id', Integer, ForeignKey('papers.id'), primary_key=True),
    Column('label_id', Integer, ForeignKey('labels.id'), primary_key=True)
)


class Paper(Base):
    """Paper model to store arXiv paper information"""
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    arxiv_id = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    authors = Column(Text)  # JSON string of authors list
    summary = Column(Text)  # Original abstract
    published = Column(String(20))
    updated = Column(String(20))
    url = Column(String(500))
    pdf_url = Column(String(500))
    categories = Column(Text)  # JSON string of categories list
    
    # Analysis results
    is_benchmark = Column(Boolean, default=False)
    benchmark_confidence = Column(String(20))  # Confidence score from local model
    full_analysis = Column(Text)  # Full analysis from large model
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to labels
    labels = relationship('Label', secondary=paper_labels, back_populates='papers')
    
    def __repr__(self):
        return f"<Paper(arxiv_id='{self.arxiv_id}', title='{self.title[:50]}...')>"


class Label(Base):
    """Label model for classification tags"""
    __tablename__ = 'labels'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50))  # Category group for the label
    description = Column(Text)
    
    # Relationship to papers
    papers = relationship('Paper', secondary=paper_labels, back_populates='labels')
    
    def __repr__(self):
        return f"<Label(name='{self.name}')>"


class AnalysisLog(Base):
    """Log table for tracking analysis history"""
    __tablename__ = 'analysis_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_arxiv_id = Column(String(50), ForeignKey('papers.arxiv_id'))
    action = Column(String(50))  # 'fetch', 'classify', 'analyze', 'label'
    status = Column(String(20))  # 'success', 'failed', 'skipped'
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AnalysisLog(paper_arxiv_id='{self.paper_arxiv_id}', action='{self.action}')>"


def get_engine():
    """Create and return database engine"""
    return create_engine(f'sqlite:///{DATABASE_PATH}', echo=False)


def get_session():
    """Create and return a new database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_database():
    """Initialize the database and create all tables"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"Database initialized at: {DATABASE_PATH}")
    return engine


def init_labels(session, labels_list, label_categories):
    """Initialize labels in the database"""
    from config import CLASSIFICATION_LABELS, LABEL_CATEGORIES
    
    # Create reverse mapping from label to category
    label_to_category = {}
    for category, labels in label_categories.items():
        for label in labels:
            label_to_category[label] = category
    
    for label_name in labels_list:
        # Check if label already exists
        existing = session.query(Label).filter_by(name=label_name).first()
        if not existing:
            category = label_to_category.get(label_name, "其他")
            new_label = Label(name=label_name, category=category)
            session.add(new_label)
    
    session.commit()
    print(f"Labels initialized: {len(labels_list)} labels")


if __name__ == "__main__":
    # Initialize database when run directly
    from config import CLASSIFICATION_LABELS, LABEL_CATEGORIES
    init_database()
    session = get_session()
    init_labels(session, CLASSIFICATION_LABELS, LABEL_CATEGORIES)
    session.close()
    print("Database setup complete!")
