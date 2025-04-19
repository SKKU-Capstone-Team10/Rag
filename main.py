from fastapi import FastAPI

from api.router import api_router

app = FastAPI()

@app.get('/') # Greeting Page
def root():
    return {'message': "Capstone Team 10 Rag"}

# Include api routers
app.include_router(api_router, prefix='/api')