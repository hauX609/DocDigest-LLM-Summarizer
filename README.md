# Doc Digest LLM Summarizer

An interactive Streamlit app to analyze and summarize PDF documents using advanced Transformer models via Hugging Face and LangChain.

##  Features

-  Upload and summarize multiple PDF files
-  Choose **Abstractive** or **Extractive** summarization
-  Choose from 3 powerful models: LaMini-Flan-T5, BART, Pegasus
-  Visual document analysis: word count, reading time, vocabulary diversity
-  Download the summary
-  GPU acceleration (if available)

##  Supported Models

| Model Name              | Model Path on Hugging Face      |
|-------------------------|----------------------------------|
| LaMini-Flan-T5-248M     | `MBZUAI/LaMini-Flan-T5-248M`     |
| BART Large CNN          | `facebook/bart-large-cnn`        |
| Pegasus XSum            | `google/pegasus-xsum`            |

##  Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/document-summarizer.git
cd document-summarizer
pip install -r requirements.txt
