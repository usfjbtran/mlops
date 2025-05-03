from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

class SubredditRequest(BaseModel):
    subreddit: str
    limit: int = 10

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_subreddit(request: SubredditRequest):
    url = f"https://www.reddit.com/r/{request.subreddit}/top.json?limit={request.limit}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for post in data["data"]["children"]:
            post_data = post["data"]
            posts.append({
                "title": post_data["title"],
                "score": post_data["score"],
                "url": post_data["url"]
            })
        
        return {"subreddit": request.subreddit, "posts": posts}
    except Exception as e:
        return {"error": str(e)} 