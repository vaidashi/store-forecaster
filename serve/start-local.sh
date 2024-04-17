export PYTHONPATH=$PYTHONPATH:$PWD
# be in serve directory
uvicorn app:app --host 127.0.0.1 --port 8000 --reload