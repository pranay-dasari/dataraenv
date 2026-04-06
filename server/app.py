import uvicorn
from datara_env.server import app  # Your existing FastAPI app

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Use 'app' directly, not string

if __name__ == "__main__":
    main()