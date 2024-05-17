from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import re
import uvicorn
from playwright.async_api import async_playwright
import spacy
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
import os
from typing import List
import cohere
from urllib.parse import urlparse
import subprocess
from typing import Dict
from typing import Optional
import json

# Import the necessary module for form parsing
from fastapi import Form

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Spacy and other libraries
nlp = spacy.load("en_core_web_lg")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
vectorizer = TfidfVectorizer()

genai.configure(api_key="AIzaSyA3PnUuCqoZ2vxfbTyv0Ii6SQTwhlaCtc0")  # Replace with your actual API key
co = cohere.Client(api_key="Dm47uGZGXLiw9KllCvk46WadVIavmosByA6xXiwq")

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
}

class AnalysisRequest(BaseModel):
    url_or_keywords: str
    brand_voice: Optional[str] = None
    include_performance_analysis: Optional[bool] = False

@app.on_event("startup")
async def startup_event():
    global browser, context
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    # Set user agent at context level
    context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")

@app.on_event("shutdown")
async def shutdown_event():
    await context.close()
    await browser.close()

@app.get("/", response_class=HTMLResponse)
async def get_template(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def clean_text(text):
    # This regex will replace any brackets, double quotes, and single quotes with nothing (effectively removing them)
    return re.sub(r'[^a-zA-Z\s\'-]', '', text)

# Define manual stopwords globally
manual_stopwords = {
    'www', 'com', 'org', 'net', 'io', 'co', 'http', 'https', 'price', 'feature', 'benefit', 'sale',
    'today', 'late', 'tomorrow', 'member', 'subscription', 'free', 'shipping', 'order', 'filter',
    'quick', 'new', 'style', 'sure', 'right', 'join', 'club', 'facebook', 'content', 'add', 'make', 
    'update', 'month', 'usd', '$', 'size', 'select', 'prepare', 'brand', 'pages', 'the', 'for', 'shop', 
    'sell', 'pricefrom', 'price', 'sign', 'bud', 'taste', 'collaboration', 'request', 'multi', 'tog',
    'and', 'Us', 'US', 'cart', 'policy', 'tax', 'sales', 'view', 'our', 'website', 'site', 'gift','collection','shop','Shop','nicu','wholesale','shop','shopping','cookies','Shopping'
}

async def generate_themes(full_domain, page_text, meta_tags):
    message = f"""
    Analyze the following website data:
    Full domain: {full_domain}
    Page text: {page_text}
    Meta Content: {meta_tags}
    First, identify the brand name and insert it into the themes[] list as the first entry. Capitalize all brand name words and make sure neither are in all caps. Then, identify the top 5 themes and insert them into the themes[] list. Keywords in the meta tags list should be assigned higher weight than page text terms. Avoid using abbreviations: for example, if the domain is AISchool.com and the content says Atlanta International School, list Atlanta International School instead of AI School. Print brand name and themes like this: themes: ['brand name','theme1','theme2','theme3',etc] """
    
    response = co.chat(model="command-r-plus", message=message)
    if response:
        try:
            themes_extract = re.search(r'themes:\s*\[(.*?)\]', response.text).group(1).replace("'", "")
            themes = themes_extract.split(", ")
            return themes
        except AttributeError as e:
            logging.error(f"Regex parsing failed: {str(e)}")
            return None
    else:
        logging.error("No response from Cohere API.")
        return None

def run_lighthouse(url: str, device: str) -> Dict[str, float]:
    try:
        # Run Lighthouse command
        result = subprocess.run(['lighthouse', url, '--output=json', '--only-categories=performance', '--emulated-form-factor=' + device], capture_output=True)
        # Parse Lighthouse JSON output
        lighthouse_output = json.loads(result.stdout)
        # Extract performance score
        performance_score = lighthouse_output['categories']['performance']['score']
        return {'performance_score': performance_score}
    except Exception as e:
        # Log the error
        logging.error(f"Error in running Lighthouse for URL: {url}, device: {device}. Error: {e}")
        # Return a default score or an empty dictionary
        return {'performance_score': 0.0}  # You can adjust the default score as needed

@app.get("/lighthouse")
async def get_lighthouse_scores(url: str):
    try:
        # Run Lighthouse for desktop score
        desktop_score = run_lighthouse(url, 'desktop')
        # Run Lighthouse for mobile score
        mobile_score = run_lighthouse(url, 'mobile')
        # Return the scores
        return JSONResponse(content={'desktop_score': desktop_score, 'mobile_score': mobile_score})
    except Exception as e:
        # Log the error
        logging.error(f"Error in getting Lighthouse scores for URL: {url}. Error: {e}")
        # Return an error response
        return JSONResponse(content={'error': 'Failed to get Lighthouse scores'}, status_code=500)

async def scrape(url: str):
    try:
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")

        scraped_data = await page.evaluate('''() => {
            const content = [];
            const footer = document.querySelector('footer');
            const selectors = ['h1', 'h2', 'h3', 'h4', 'p', 'div', 'span', 'li', 'img'];

            const metaTags = Array.from(document.querySelectorAll('meta')).filter(meta => {
                const prop = meta.getAttribute('property') || meta.getAttribute('name');
                return prop && (prop.toLowerCase().includes('title') || prop.toLowerCase().includes('description') || prop.toLowerCase().includes('keywords'));
            });

            const titleContent = document.querySelector('title') ? document.querySelector('title').innerText : '';
            const metaContent = metaTags.map(meta => meta.getAttribute('content')).filter(content => content).join(' ') + ' ' + titleContent;

            selectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(element => {
                    if (!footer || !footer.contains(element)) {
                        const computedStyle = window.getComputedStyle(element);
                        if (computedStyle.display !== 'none' && computedStyle.visibility !== 'hidden') {
                            content.push(element.innerText.trim() + ' ');
                        }
                    }
                });
            });

            return {
                'meta_tags': metaTags.map(meta => ({
                    name: meta.getAttribute('name') || meta.getAttribute('property'),
                    content: meta.getAttribute('content')
                })),
                'page_text': content.join(' ')
            };
        }''')

        # Check for content presence and process accordingly
        if scraped_data['meta_tags'] or scraped_data['page_text']:
            filtered_text = filter_text(scraped_data['page_text'])

            return_data = {
                'full_domain': url,
                'meta_tags': scraped_data['meta_tags'],
                'page_text': filtered_text,
            }

            print("Full Domain:", return_data['full_domain'])
            print("Meta Tags:", return_data['meta_tags'])
            print("Page Text:", return_data['page_text'])

            return return_data
        else:
            raise ValueError("No content or meta tags found on the page.")

    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail="Scraping failed")


def filter_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove any unbroken string with any special characters
    text = re.sub(r'\S*[^A-Za-z0-9\s]\S*', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert text to lowercase to ensure uniqueness is case-insensitive
    words = text.lower().split()

    # Use a set to avoid duplicates
    unique_words = set()

    # Filter and collect unique words that are not stopwords and are purely alphabetical
    filtered_words = []
    for word in words:
        if word not in manual_stopwords and word.isalpha() and word not in unique_words:
            filtered_words.append(word)
            unique_words.add(word)

    return ' '.join(filtered_words)


async def generate_ad_copy(brand_name, themes, brand_voice):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",)
        prompt = f"""
        Here's information for the following prompt: 
- You are a marketing copywriter.  
- You are writing for this company: {brand_name}
- You're writing ads using these themes: {themes}

High quality advertising copy consists of four parts, following the AIDA Framework. 1 - Attention 2 - Interest 3 - Desire 4 - Action 

Definition of good Google Ads responsive search ads - A responsive search ad is comprised of 15 headlines and 4 descriptions - Keep in mind that headlines and descriptions may appear in any order. - Try writing each headline as if they’ll appear together in your ad. - Be sure to include at least one of your themes in your headlines, and create headlines that are relevant to the themes you’re targeting. - Try highlighting additional product or service benefits, a problem you’re solving, or shipping and returns information. Learn more about Creating effective Search ads. - You can provide even more headlines by creating variations of the headlines you’ve already entered. For example, try a different call to action. - Craft messaging that focuses on user benefits. Users respond to ads that speak to their needs. Tie your headline and description line’s messaging to your keywords. Users tend to engage with ads that appear most relevant to their search. Avoid generic language in your ads. Use specific calls to action. Generic calls to action often show decreased engagement with ads. Language / creative instructions: - Do not repeat themes or phrases. Each headline and description should communicate a unique and compelling benefit or product feature. Avoid using too many "?" or "!" or "-" or ":". Some are OK. Don't use the brand name too many times.

        1. Write 15 ad copy lines that are 5-6 tokens in length (no more than 30 total characters). Capitalize the first letter of every word. Insert them into a list called ad_headlines. Use strong calls-to-action(CTA) and focus on grabbing attention with product benefits. End each headline with a "/"
        2. Create a new line and write 4 ad copy lines that are 17-18.2 tokens in length (no more than 90 total characters) and insert them into a list called ad_descriptions. End each headline with a "/"

        Write in this style: {brand_voice}.

        The final output should be 2 python lists like this:
        ad_headlines = [headline, headline, headline, etc...15 total]
        ad_descriptions = [description, description, description, etc...4 total]"""
        response = model.generate_content(prompt)
        print(response)
        if response:
            if hasattr(response, 'text'):
                ad_copy = response.text.replace("```python","").replace("```","")
                print(ad_copy)
                return ad_copy
        else:
            logging.error("No response from the generative model.")
            return []
    except Exception as e:
        logging.error(f"Error in generating ad copy: {str(e)}")
        return []
    
async def generate_ad_copy_keywords(keywords, brand_voice):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        prompt = f"""
        Here's information for the following prompt: 
- You're writing ads using these themes: {keywords}

High quality advertising copy consists of four parts, following the AIDA Framework. 1 - Attention 2 - Interest 3 - Desire 4 - Action 

Definition of good Google Ads responsive search ads - A responsive search ad is comprised of 15 headlines and 4 descriptions - Keep in mind that headlines and descriptions may appear in any order. - Try writing each headline as if they’ll appear together in your ad. - Be sure to include at least one of your themes in your headlines, and create headlines that are relevant to the themes you’re targeting. - Try highlighting additional product or service benefits, a problem you’re solving, or shipping and returns information. Learn more about Creating effective Search ads. - You can provide even more headlines by creating variations of the headlines you’ve already entered. For example, try a different call to action. - Craft messaging that focuses on user benefits. Users respond to ads that speak to their needs. Tie your headline and description line’s messaging to your keywords. Users tend to engage with ads that appear most relevant to their search. Avoid generic language in your ads. Use specific calls to action. Generic calls to action often show decreased engagement with ads. Language / creative instructions: - Do not repeat themes or phrases. Each headline and description should communicate a unique and compelling benefit or product feature. Avoid using too many "?" or "!" or "-" or ":". Some are OK. Don't use the brand name too many times.

        1. Write 15 ad copy lines that are 5-6 tokens in length (no more than 30 total characters). Capitalize the first letter of every word. Insert them into a list called ad_headlines. Use strong calls-to-action(CTA) and focus on grabbing attention with product benefits. End each headline with a "/"
        2. Create a new line and write 4 ad copy lines that are 17-18.2 tokens in length (no more than 90 total characters) and insert them into a list called ad_descriptions. End each headline with a "/"

        Write in this style: {brand_voice}.

        The final output should be 2 python lists like this:
        ad_headlines = [headline, headline, headline, etc...15 total]
        ad_descriptions = [description, description, description, etc...4 total]
"""
        response = model.generate_content(prompt)
        if response:
            if hasattr(response, 'text'):
                ad_copy = response.text.replace("```python","").replace("```","")
                print(ad_copy)
                return ad_copy
        else:
            logging.error("No response from the generative model.")
            return []
    except Exception as e:
        logging.error(f"Error in generating ad copy: {str(e)}")
        return []


@app.post("/analyze")
async def analyze(data: AnalysisRequest):
    url_or_keywords = data.url_or_keywords
    brand_voice = data.brand_voice
    include_performance_analysis = data.include_performance_analysis
    try:
        themes = None
        ad_copy = None
        performance_results = {}  # Initialize performance_results dictionary

        if url_or_keywords.startswith(("http://", "https://")):
            # Scraping and processing content
            scraped_data = await scrape(url_or_keywords)

            # Check if there's meaningful content to process
            if not scraped_data['meta_tags'] and not scraped_data['page_text']:
                logging.error("No content to process after scraping.")
                raise ValueError("No content available from the scrape.")

            # Generate themes
            themes = await generate_themes(scraped_data['full_domain'], scraped_data['page_text'], [tag['content'] for tag in scraped_data['meta_tags']])

            if not themes:
                logging.error("Failed to generate themes.")
                raise ValueError("Theme generation failed.")

            # Generate ad copy
            ad_copy = await generate_ad_copy(themes[0], themes[1:], brand_voice)

            if not ad_copy:
                logging.error("Ad copy generation failed.")
                raise ValueError("Ad copy generation failed.")
            
            if include_performance_analysis:
                # Run page performance analysis
                performance_results = run_lighthouse(url_or_keywords, 'desktop')  # Assuming desktop analysis

        else:
            # Using keywords directly as themes
            keywords = [keyword.strip() for keyword in url_or_keywords.split(',')]
            themes = keywords  
            ad_copy = await generate_ad_copy_keywords(keywords, brand_voice)

        response_content = {
            "top_themes_text": themes,
            "ad_copy": ad_copy,
            "performance_results": performance_results  # Include performance results in the response
        }
        return JSONResponse(content=response_content, status_code=status.HTTP_200_OK)

    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Sorry, this site could not be scraped. Some sites block services like this. Try manually entering keywords instead."}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)