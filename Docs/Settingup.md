The actual application entry point is app/main.py. This is where the FastAPI instance (app = FastAPI(...)) is created and where your routes are included.

server.py: This is a convenience script for local development. It loads your .env file and starts uvicorn. It points to app.main:app, which confirms that the real app lives in app/main.py.
app/main.py: This is the core of your server. In a production environment (like Docker/Cloud Run), you should run this directly.

## step 1 
created .dockerignore

## step 2
created then Dockerfile

## step 3
# Build the image
docker build -t wekraft-agent .

# Run it (mounts your .env for testing)
docker run -p 8080:8080 --env-file .env wekraft-agent

## step 4


## After deployment (important!)

Go to Google Cloud Console → Cloud Run → your service → Edit & Deploy New Revision
Under Environment variables, add everything from your .env file (OpenAI keys, LangSmith, etc.)
Never put secrets in the Docker image.

That’s it! You now have a production-ready agent on Cloud Run.
