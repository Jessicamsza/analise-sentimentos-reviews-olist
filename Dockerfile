
FROM python:3.9-slim


WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Expõe a porta que o Streamlit usa por padrão.
EXPOSE 8501


CMD ["streamlit", "run", "app/app.py"]