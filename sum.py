import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import base64
import time
import os
from tempfile import mkstemp
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


st.set_page_config(layout="wide", page_title="Advanced Document Summarization App")

MODELS = {
    "LaMini-Flan-T5-248M": "MBZUAI/LaMini-Flan-T5-248M",
    "BART-Large-CNN": "facebook/bart-large-cnn",
    "Pegasus-XSum": "google/pegasus-xsum"
}

@st.cache_resource(show_spinner="Loading base model...")
def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    if torch.cuda.is_available():
        model = model.to('cuda')
    return tokenizer, model

base_tokenizer, base_model = load_base_model()

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return tokenizer, model

def is_valid_pdf(file):
    try:
        header = file.read(4)
        file.seek(0)
        return header == b'%PDF'
    except:
        return False

def file_preprocessing(file, chunk_size=512, chunk_overlap=50):
    try:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(pages)
        return " ".join([text.page_content for text in texts])
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def analyze_document(text):
    words = text.split()
    word_count = len(words)
    word_freq = Counter(words)
    return {
        'word_count': word_count,
        'page_count': text.count('\f') + 1 if '\f' in text else 1,
        'reading_time': max(1, round(word_count / 200)),
        'vocab_size': len(word_freq),
        'unique_ratio': round(len(word_freq) / word_count * 100, 2) if word_count > 0 else 0,
        'common_words': word_freq.most_common(5)
    }

def plot_analysis(analysis):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(['Words', 'Pages', 'Reading Time'], 
             [analysis['word_count'], analysis['page_count'], analysis['reading_time']],
             color=['skyblue', 'lightgreen', 'salmon'])
    ax[0].set_title('Document Metrics')
    
    ax[1].pie([analysis['unique_ratio'], 100 - analysis['unique_ratio']],
             labels=['Unique Words', 'Repeated Words'],
             autopct='%1.1f%%')
    ax[1].set_title('Vocabulary Diversity')
    
    st.pyplot(fig)

def secure_file_handling(uploaded_file):
    try:
        if not is_valid_pdf(uploaded_file):
            st.error("Invalid PDF file")
            return None
        fd, path = mkstemp(suffix='.pdf')
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(uploaded_file.getbuffer())
        return path
    except Exception as e:
        st.error(f"File handling error: {str(e)}")
        return None

def llm_pipeline(filepath, summary_length='Medium', summary_type='Abstractive', selected_model="MBZUAI/LaMini-Flan-T5-248M"):
    try:
        length_map = {
            'Short': {'max_length': 150, 'min_length': 30},
            'Medium': {'max_length': 300, 'min_length': 50},
            'Long': {'max_length': 500, 'min_length': 100}
        }
        
        input_text = file_preprocessing(filepath)
        if not input_text:
            return None

        analysis = analyze_document(input_text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        documents = text_splitter.create_documents([input_text])

        if summary_type == 'Extractive':
            pipe = pipeline(
                "text2text-generation",
                model=base_model,
                tokenizer=base_tokenizer,
                clean_up_tokenization_spaces=True,
                **length_map[summary_length]
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
            result = summarize_chain.invoke(documents)
            summary = result['output_text'] if isinstance(result, dict) and 'output_text' in result else str(result)

        else:
            tokenizer, model = (
                (base_tokenizer, base_model)
                if selected_model == "MBZUAI/LaMini-Flan-T5-248M"
                else load_model(MODELS[selected_model])
            )
            pipe_sum = pipeline(
                'summarization',
                model=model,
                tokenizer=tokenizer,
                clean_up_tokenization_spaces=True,
                **length_map[summary_length]
            )

            summary_parts = []
            for doc in documents:
                chunk = doc.page_content
                tokenized_input=tokenizer(chunk, return_tensors="pt", truncation=False)
                input_len = tokenized_input['input_ids'].shape[1]
                max_pos = getattr(model.config, 'max_position_embeddings', 512) 
                if input_len <= max_pos:
                    dynamic_max = min(length_map[summary_length]['max_length'], max(30, input_len // 2))
                    dynamic_min = min(length_map[summary_length]['min_length'], dynamic_max  - 10)
                    try:
                        result = pipe_sum(chunk, max_length=dynamic_max, min_length=dynamic_min, clean_up_tokenization_spaces=True)
                        summary_parts.append(result[0]['summary_text'])
                    except Exception as e:
                        st.warning(f"Skipped a chunk due to summarization error: {e}")
                else:
                    st.warning("Skipped a chunk that exceeds model token limit.")
            summary = " ".join(summary_parts)

        return {
            'summary': summary,
            'analysis': analysis
        }
    except torch.cuda.OutOfMemoryError:
        st.error("Out of memory! Try a smaller model or reduce chunk size")
        return None
    except Exception as e:
        st.error(f"Summarization failed: {str(e)}")
        return None

@st.cache_data
def displayPDF(file):
    try:
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def format_summary(summary, display_option):
    if isinstance(summary, dict) and 'summary_text' in summary:
         summary = summary.get('summary_text') or summary.get('output_text') or str(summary)
    elif hasattr(summary, 'page_content'):  # LangChain Document
        summary = summary.page_content
    elif isinstance(summary, list) and isinstance(summary[0], dict) and 'summary_text' in summary[0]:
        summary = " ".join([s['summary_text'] for s in summary])
        
    if not isinstance(summary, str):
        return "Invalid summary format."

    if display_option == 'Bullet Points':
        sentences = summary.split('. ')
        return "\n".join([f"• {sentence.strip()}" for sentence in sentences if sentence.strip()])
    elif display_option == 'Key Sentences':
        sentences = summary.split('. ')
        return ". ".join(sentences[:3]) + "."
    return summary

def main():
    st.title("📄 Advanced Document Summarization App")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        selected_model = st.selectbox("Choose Model", list(MODELS.keys()))
        
        with st.expander("Processing Options"):
            chunk_size = st.slider("Chunk size", 100, 1000, 200)
            chunk_overlap = st.slider("Chunk overlap", 0, 100, 50)
            
        with st.expander("Summary Options"):
            summary_length = st.select_slider("Length", ['Short', 'Medium', 'Long'], 'Medium')
            summary_type = st.radio("Type", ['Abstractive', 'Extractive'])
            display_option = st.radio("Format", ['Full Summary', 'Bullet Points', 'Key Sentences'])
        
        with st.expander("Performance"):
            if torch.cuda.is_available():
                st.success("GPU Available")
            else:
                st.warning("Using CPU")

    uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Summarize", type="primary"):
            for uploaded_file in uploaded_files:
                with st.expander(f"Processing: {uploaded_file.name}", expanded=True):
                    if uploaded_file.size == 0:
                        st.error("Empty file skipped")
                        continue
                        
                    if uploaded_file.size > 50 * 1024 * 1024:
                        st.error("File too large (max 50MB)")
                        continue
                        
                    with st.spinner("Processing..."):
                        filepath = secure_file_handling(uploaded_file)
                        if not filepath:
                            continue
                            
                        try:
                            with st.spinner("Analyzing document..."):
                                result = llm_pipeline(
                                    filepath,
                                    summary_length,
                                    summary_type,
                                    selected_model
                                )
                                
                            if result:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.info("📊 Document Analysis")
                                    st.write(f"📝 Words: {result['analysis']['word_count']}")
                                    st.write(f"📄 Pages: {result['analysis']['page_count']}")
                                    st.write(f"⏱️ Reading time: {result['analysis']['reading_time']} mins")
                                    plot_analysis(result['analysis'])
                                
                                with col2:
                                    st.info("📄 Original Document")
                                    displayPDF(filepath)
                                    st.info("✅ Summary")
                                    formatted = format_summary(result['summary'], display_option)
                                    st.success(formatted)
                                    st.download_button(
                                        "📥 Download Summary",
                                        formatted,
                                        f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt"
                                    )
                        finally:
                            if os.path.exists(filepath):
                                try:
                                    os.unlink(filepath)
                                except:
                                    pass

if __name__ == "__main__":
    main()
