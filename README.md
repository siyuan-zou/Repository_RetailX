## Info  
This project is for the 2-day Hackathon organized jointly by Ecole polytechnique, McKinsey and Mistral AI.  
Participants are CHEN Yiming, MONTAGNE Thibaud, ZOU Siyuan, ZOU Yuran (by name order).  

# RetailX-Hackathon

A MVP (Minimum Viable Product) Prototype of RetailX,  an intelligent AI assistant powered by [Mistral AI](https://github.com/mistralai) designed to provide an online shopping experience comparable to in-store retail.

Key ideas :  
- User-friendly, natural, visible shopping experience;
- Customized guidance, following the customer's previous choices;
- Scalable and comprehensible structure.

## Demo picture

<img src="Demo/demopic.png" alt="2" style="zoom: 33%;" />



## Usage

0. Create a conda or python venv

```bash
conda create -n retailx python=3.10.15

conda activate retailx
```

1. Install the dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```

2. Download the dataset [Amazon-Sales-Dataset-EDA-94](https://www.kaggle.com/code/sonawanelalitsunil/amazon-sales-dataset-eda-94/input) and copy it to the root folder of the project.

3. Create a file named "env.py" and add the following items:

```python
file_path = 'Data/amazon.csv'  # by default
api_key = "your_mistral_AI_API_key"
history_file_path = 'history.txt' # by default

pricer_path = 'Personalities/personality_pricer.txt' 
discounter_path = 'Personalities/personality_discounter.txt'
rater_path = 'Personalities/personality_rater.txt'
salesperson_path = 'Personalities/salesperson.txt'
keyword_extractor_path = 'Personalities/keyword.txt'
conselor_path='Personalities/conselor.txt'

```

**ATTENTION: Tested Python version: `3.10.15`**

4. Execute in the system terminal `gradio app.py`, then a link directing to the agent appears in the terminal.

The default link is http://127.0.0.1:7860

Note: Initial Data Preparation may take 30s to 1min.


## File Description

Frontend

- `app.py`: The main application file that runs the Gradio interface.
- `style.css`: Contains custom CSS styles for the Gradio interface.
- `api_frontend.py`: The Interface to interact with the backend.

Backend

- `api_backend.py`: API for the frontend provided by the backend, also the local test code for backend.
- `functions.py`: Functions implementing LLM calls.
- `embed.py`: Functions calculating the objects' embeddings
- `api_backend.py`: API for the frontend provided by the backend, also the local test code for backend. Contains the main workflow of Backend.  
- `functions.py`: Auxialiary Functions, implementing embedding comparisons and LLM calls.  

Data

- `env.py`: Contains environment variables such as file paths and API keys.
- `/Data/amazon.csv`: The dataset file containing Amazon sales data.

Miscellaneous

- `requirements.txt`: Lists all the dependencies required to run the project.
- `README.md`: This file, providing an overview and instructions for the project.
- `mock_request.py`: mock data for the frontend.  
- `history.txt`: A file to store the history of processed data or interactions. Used when testing backend.  
- `Personalities`: containing System Prompts of all LLM's used in the project.

## Technical Details

### Front-End  

The frontend is powered by the framework `gradio`. The backend is a simplified synchronous implementation of the system. Instead of leveraging asynchronous programming for handling concurrent requests or external API calls, it processes all incoming requests in a blocking manner.   

### Back-End  
The backend workflow chart looks like this: 

<img src="Demo/chart.png" alt="2" style="zoom: 25%;" />


We use multiple Mistral AI LLM personalities:  
The historian extracts a one-phrase description of the user's demand from chat history;  
The price/discount/rating filterer extracts any "hard" criteria on those numerical columns, if applicable;  
The salesperson generates (based on previous conversations and relevant objects) a pertinent follow-up question to help narrow down.  
The suggestionner suggests (based on most relevant items) some items in another category as possible cross-sales.

The actual object-matching is done by local transformer models, by comparing text embeddings.  

Cross-sales is done by randomly picking marchandizes from the categories that appear among the most-relevant articles.  
We have hard-filter identification mechanisms on numerical features(price,discount rate,rating) and this is easily scalable if there is more similar features.  

## Model Evaluation

### Good points  
Mistral AI LLMs generally perform quite well on the given tasks;  
Rare hallucinations are observed, even with Large, but detectable and fixable by asking LLM to re-run.

Our chatbot is capable of treating common merchandies search requests, and can raise pertinent questions to narrow down.  
The suggestion engine successfully proposes the most relevant items, in a user-friendly way.  

### Not-yet-good points  
Unconventional (not following guidance) user inputs are not always well treated;  
Detailed comparison between different encoders or LLM models are not carried out;  
There should be a more flexible way to treat 'hard' user-demand-filters, in case there are more numerical columns;  
Cross-sales sometimes gives same-category objects because there are not any closely-related category in the database. Scaling up the database may help a lot, but this is not experimented. 
If we want in the same time top items AND cross-sales, the reaction time is too long. (Too many LLM requests to do).   
## Future possibilities  

A (not implemented but) potentially promising idea is to train a LLM coder to write and auto-execute code snippets that can operate autonomously on the database;  
This can allow for more versatile question-answering capabilities related to the database.  

A Graph Neural Network that defines item similarity might be able to give more logical cross-sales suggestions.

