"""
Visualization Dashboard for arXiv Benchmark Papers

A Streamlit-based web application for visualizing benchmark paper analysis results.
Features:
- Interactive label selection
- Label intersection matrix (heatmap)
- Paper browsing and filtering
- Statistics dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

from config import CLASSIFICATION_LABELS, LABEL_CATEGORIES, DATABASE_PATH
from database import get_session, Paper, Label


def get_all_labels():
    """Get all labels from database"""
    session = get_session()
    labels = session.query(Label).all()
    session.close()
    return [(l.name, l.category) for l in labels]


def get_papers_with_labels(selected_labels=None):
    """Get papers with optional label filter"""
    session = get_session()
    
    query = session.query(Paper).filter_by(is_benchmark=True)
    
    if selected_labels:
        # Filter papers that have ALL selected labels
        for label_name in selected_labels:
            query = query.filter(
                Paper.labels.any(Label.name == label_name)
            )
    
    papers = query.all()
    
    result = []
    for p in papers:
        result.append({
            'arxiv_id': p.arxiv_id,
            'title': p.title,
            'authors': json.loads(p.authors) if p.authors else [],
            'summary': p.summary,
            'published': p.published,
            'url': p.url,
            'categories': json.loads(p.categories) if p.categories else [],
            'labels': [l.name for l in p.labels],
            'confidence': p.benchmark_confidence,
            'analysis': p.full_analysis
        })
    
    session.close()
    return result


def get_label_intersection_matrix(selected_labels=None):
    """Calculate intersection matrix between labels"""
    session = get_session()
    
    if selected_labels:
        labels = session.query(Label).filter(Label.name.in_(selected_labels)).all()
    else:
        labels = session.query(Label).all()
    
    label_names = [l.name for l in labels]
    n = len(labels)
    
    # Initialize matrix
    matrix = [[0] * n for _ in range(n)]
    
    # Calculate intersections
    for i, label1 in enumerate(labels):
        papers1 = set(p.id for p in label1.papers)
        for j, label2 in enumerate(labels):
            papers2 = set(p.id for p in label2.papers)
            matrix[i][j] = len(papers1 & papers2)
    
    session.close()
    
    return pd.DataFrame(matrix, index=label_names, columns=label_names)


def get_statistics():
    """Get overall statistics"""
    session = get_session()
    
    total_papers = session.query(Paper).count()
    benchmark_papers = session.query(Paper).filter_by(is_benchmark=True).count()
    
    # Label distribution
    labels = session.query(Label).all()
    label_counts = {l.name: len(l.papers) for l in labels}
    
    # Category distribution
    category_counts = {}
    for l in labels:
        cat = l.category or "Other"
        if cat not in category_counts:
            category_counts[cat] = 0
        category_counts[cat] += len(l.papers)
    
    session.close()
    
    return {
        'total_papers': total_papers,
        'benchmark_papers': benchmark_papers,
        'label_counts': label_counts,
        'category_counts': category_counts
    }


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="arXiv Benchmark Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 arXiv Benchmark Paper Analyzer")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📈 Statistics Dashboard", "🏷️ Label Matrix", "📚 Browse Papers"]
    )
    
    if page == "📈 Statistics Dashboard":
        show_statistics_page()
    elif page == "🏷️ Label Matrix":
        show_label_matrix_page()
    elif page == "📚 Browse Papers":
        show_papers_page()


def show_statistics_page():
    """Statistics dashboard page"""
    st.header("📈 Statistics Dashboard")
    
    stats = get_statistics()
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Papers", stats['total_papers'])
    with col2:
        st.metric("Benchmark Papers", stats['benchmark_papers'])
    with col3:
        if stats['total_papers'] > 0:
            ratio = stats['benchmark_papers'] / stats['total_papers'] * 100
            st.metric("Benchmark Ratio", f"{ratio:.1f}%")
        else:
            st.metric("Benchmark Ratio", "N/A")
    
    st.markdown("---")
    
    # Label distribution chart
    st.subheader("Label Distribution")
    
    label_counts = stats['label_counts']
    # Filter to only show labels with papers
    active_labels = {k: v for k, v in label_counts.items() if v > 0}
    
    if active_labels:
        df = pd.DataFrame({
            'Label': list(active_labels.keys()),
            'Count': list(active_labels.values())
        })
        df = df.sort_values('Count', ascending=True)
        
        fig = px.bar(
            df,
            x='Count',
            y='Label',
            orientation='h',
            title='Papers per Label',
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=max(400, len(active_labels) * 25))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No papers with labels found. Run the analyzer first.")
    
    # Category distribution
    st.subheader("Category Distribution")
    
    cat_counts = stats['category_counts']
    active_cats = {k: v for k, v in cat_counts.items() if v > 0}
    
    if active_cats:
        fig = px.pie(
            values=list(active_cats.values()),
            names=list(active_cats.keys()),
            title='Papers by Category'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_label_matrix_page():
    """Label intersection matrix page"""
    st.header("🏷️ Label Intersection Matrix")
    
    st.markdown("""
    This matrix shows the number of papers that have both row and column labels.
    Select specific labels to focus on particular intersections.
    """)
    
    # Label selection
    st.subheader("Select Labels")
    
    # Group labels by category
    for category, labels in LABEL_CATEGORIES.items():
        st.markdown(f"**{category}:**")
        cols = st.columns(4)
        for i, label in enumerate(labels):
            with cols[i % 4]:
                st.checkbox(label, key=f"label_{label}")
    
    # Get selected labels
    selected_labels = [
        label for label in CLASSIFICATION_LABELS
        if st.session_state.get(f"label_{label}", False)
    ]
    
    # Quick selection buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Select All"):
            for label in CLASSIFICATION_LABELS:
                st.session_state[f"label_{label}"] = True
            st.rerun()
    with col2:
        if st.button("Clear All"):
            for label in CLASSIFICATION_LABELS:
                st.session_state[f"label_{label}"] = False
            st.rerun()
    with col3:
        if st.button("Select Top 10 Active"):
            stats = get_statistics()
            top_labels = sorted(
                stats['label_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for label in CLASSIFICATION_LABELS:
                st.session_state[f"label_{label}"] = label in [l[0] for l in top_labels]
            st.rerun()
    
    st.markdown("---")
    
    # Generate matrix
    if selected_labels:
        st.subheader(f"Intersection Matrix ({len(selected_labels)} labels)")
        
        matrix_df = get_label_intersection_matrix(selected_labels)
        
        # Create heatmap
        fig = px.imshow(
            matrix_df,
            labels=dict(x="Label", y="Label", color="Papers"),
            x=matrix_df.columns,
            y=matrix_df.index,
            color_continuous_scale="Blues",
            aspect="auto"
        )
        
        # Add text annotations
        fig.update_traces(
            text=matrix_df.values,
            texttemplate="%{text}",
            textfont={"size": 10}
        )
        
        fig.update_layout(
            title="Label Intersection Matrix",
            height=max(500, len(selected_labels) * 40),
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show as table
        st.subheader("Matrix Data")
        st.dataframe(matrix_df, use_container_width=True)
        
        # Download option
        csv = matrix_df.to_csv()
        st.download_button(
            label="📥 Download Matrix as CSV",
            data=csv,
            file_name="label_intersection_matrix.csv",
            mime="text/csv"
        )
    else:
        st.info("Please select at least one label to view the intersection matrix.")


def show_papers_page():
    """Browse papers page"""
    st.header("📚 Browse Papers")
    
    # Filter options
    st.subheader("Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Label filter
        selected_label_filter = st.multiselect(
            "Filter by Labels",
            options=CLASSIFICATION_LABELS,
            default=[]
        )
    
    with col2:
        # Sort option
        sort_by = st.selectbox(
            "Sort By",
            ["Published Date (Newest)", "Published Date (Oldest)", "Title"]
        )
    
    # Get papers
    papers = get_papers_with_labels(selected_label_filter if selected_label_filter else None)
    
    # Sort papers
    if sort_by == "Published Date (Newest)":
        papers.sort(key=lambda x: x['published'], reverse=True)
    elif sort_by == "Published Date (Oldest)":
        papers.sort(key=lambda x: x['published'])
    else:
        papers.sort(key=lambda x: x['title'])
    
    st.markdown(f"**Found {len(papers)} papers**")
    st.markdown("---")
    
    # Display papers
    for i, paper in enumerate(papers):
        with st.expander(f"📄 {paper['title']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**arXiv ID:** {paper['arxiv_id']}")
                st.markdown(f"**Published:** {paper['published']}")
                st.markdown(f"**Authors:** {', '.join(paper['authors'][:5])}")
                if len(paper['authors']) > 5:
                    st.markdown(f"*... and {len(paper['authors']) - 5} more*")
                st.markdown(f"**URL:** [{paper['url']}]({paper['url']})")
            
            with col2:
                st.markdown(f"**Confidence:** {paper['confidence']}")
                st.markdown(f"**Categories:** {', '.join(paper['categories'])}")
            
            st.markdown("---")
            
            # Labels
            if paper['labels']:
                st.markdown("**Labels:**")
                label_cols = st.columns(4)
                for j, label in enumerate(paper['labels']):
                    with label_cols[j % 4]:
                        st.markdown(f"🏷️ {label}")
            
            # Summary
            st.markdown("**Summary:**")
            st.markdown(paper['summary'])
            
            # Analysis
            if paper['analysis']:
                st.markdown("---")
                st.markdown("**Full Analysis:**")
                st.markdown(paper['analysis'])


if __name__ == "__main__":
    main()
