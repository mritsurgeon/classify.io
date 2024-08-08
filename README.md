# Classify.io
Data Classification Tool ( For Backup TBD )

This project provides a tool to parse Veeam backup data or any data path using Apache Tika and perform Named Entity Recognition (NER) using the GLiNER model via spaCy. It also integrates with a ChromaDB vector database for data storage and retrieval and uses a chatbot interface powered by the Ollama API to respond to queries based on the parsed data.

### Test a Slimmed-Down Version of This Tool on Google Colab

- Google Colab is free and provides access to GPU.
- This version is a slimmed-down version designed to test the Gliner PII model and UI.

You can access the Colab notebook here:

https://colab.research.google.com/drive/1A7FeCPDuN5pT6jqEdKXqD4e0q5b6PBfk

## Features

* **Tika Integration**: Supports over a thousand file types, including PDF, DOCX, PPTX, and more, for metadata and text extraction.
* **Zero-Shot NER**: Utilizes GLiNER for identifying various PII and entity types without the need for pre-training.
* **UI and Visualization**: Uses Streamlit for an interactive UI and Plotly for data visualization.
* **Vector Database**: Stores parsed content and metadata in ChromaDB, allowing efficient querying and retrieval.
* **Chatbot Interface**: Leverages Ollama for generating responses to user queries.

## Installation

1. **Clone the Repository**:
   git clone <repository-url>
   cd <repository-name>

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed. Install required packages with:
   pip install -r requirements.txt

3. **Set Up Ollama Server**:
   * Download and install the Ollama server.
   * Pull the TinyLlama model:
     ollama pull tinyllama

4. **Install Java**:
   * Ensure Java (JDK 22) is installed on your system for Apache Tika.

## Configuration

* **Apache Tika**: Tika does not require additional configuration beyond ensuring Java is installed.
* **ChromaDB**: No specific setup required beyond ensuring the system can store and retrieve data as needed.

## Usage

1. **Run the Application**:
   streamlit run classifyio.py

2. **RAG Chatbot**: Interact with the chatbot to query your data. The chatbot uses the Ollama API to generate responses based on the data stored in ChromaDB.

3. **Set the Directory for Analysis**: Configure the directory containing the Veeam backup FLR files by editing the following line in the code:
   Directory containing the files:
   `directory = 'C:\\VeeamFLR\\'`
   You might choose more granular folder paths to enhance performance. This can be tested with a ratio of compute needed versus data to process, such as:
   `directory = 'C:\VeeamFLR\target_3c75de11\Volume1\Users\Administrator\Documents'`

## Project Stage

This project is currently in its infancy or Minimal Viable Product (MVP) stage. As a result, some features may be under development, and the implementation may evolve as feedback is gathered and the project matures. Due to the evolving nature of machine learning, changes and tweaks might be needed to ensure the best model is used. Different types of datasets might have varying performance outcomes on certain models or NER techniques, necessitating ongoing adjustments and optimizations.

## Lessons Learned

During the development of this project, several key insights were gained:

* **Model Performance**: Finding the right balance between accuracy and performance is crucial, especially when working with large datasets.
* **System Resources**: The resources required can vary significantly depending on the size of the dataset. While testing was conducted on a CPU, a GPU is recommended for handling large datasets to improve processing efficiency and performance.
* **Integration Challenges**: Combining different technologies like Apache Tika, spaCy, ChromaDB, and Ollama required attention to compatibility and configuration details. Each component's setup and performance has to be finely tuned to achieve a seamless workflow.

## Under Construction

* The API functionality is yet to be integrated.
* The Veeam backup server configuration in the UI currently holds placeholder data.

## License

This project is licensed under the MIT License. It is open for non-commercial use only. Commercialization or sale of this tool is prohibited.

## Screenshots 

![Streamlit1](https://github.com/user-attachments/assets/b5797936-79b2-41b2-8391-f08e313ae5b9)

