# 1) create isolated env (recommended)
conda create -n streamlit311 python=3.11 -y
conda activate streamlit311

# 2) install app deps
pip install -r requirements.txt

# 3) run app
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false

# 4) Visit
https://172.16.216.132:8000/user/{username}/proxy/8501/


