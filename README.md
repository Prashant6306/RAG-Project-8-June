docker build -t rag-streamlit1 .

docker run -d --name rag-app --network=host rag-streamlit1

t2-xlarge - 50GB memory
