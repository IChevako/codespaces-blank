from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
from google.oauth2 import service_account
from fastapi.responses import JSONResponse

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

genai.configure(api_key="AIzaSyDE9ejGsLF6Bz69K4pHxrZyOuOV0X375LU")  # Replace with your actual API key

class AnalysisRequest(BaseModel):
    url_or_keywords: str
    brand_voice: str  # New field for brand voice description
    
class GenericResponse(BaseModel):
    content: dict

@app.on_event("startup")
async def startup_event():
    global browser, context
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()

@app.on_event("shutdown")
async def shutdown_event():
    await context.close()
    await browser.close()

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"HTTP Exception: {exc.detail}"}
    )

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
    'and', 'Us', 'US', 'cart', 'policy', 'tax', 'sales', 'view', 'our', 'website', 'site', 'gift','collection','shop','Shop','nicu','wholesale','shop',' ','save'
}

async def generate_themes(full_domain, meta_content, meta_title, title_tag, page_text):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
        )

        prompt = f"""
        Here's raw data from a page scrape:
        Full domain: {full_domain}
        Meta content: {meta_content}
        Meta title: {meta_title}
        Title tag: {title_tag}
        Page text: {page_text}
        Create 2 lines like this:
        brand_name: []
        top_page_themes: []
        1. Identify the brand_name and insert it into the brand_name brackets. Capitalize both words and make sure neither are in all caps.
        2. Identify the top 5 themes and insert them into the top_page_themes brackets separated by a forward slash /.
        """
        response = model.generate_content(prompt)
        if response:
            if hasattr(response, 'text'):
                generated_text = response.text
                lines = generated_text.split('\n')
                brand_name_raw = lines[0].split('brand_name:')[1].strip() if 'brand_name:' in lines[0] else ''
                themes_text_raw = lines[1].split('top_page_themes:')[1].strip() if 'top_page_themes:' in lines[1] else ''

                # Filter out manual stopwords from brand_name and clean text
                brand_name = clean_text(' '.join(word for word in brand_name_raw.split() if word.lower() not in manual_stopwords))

                # Filter out manual stopwords from themes, clean text, and split correctly
                themes = [clean_text(theme.strip()) for theme in themes_text_raw.split('/') if theme.strip()]
                themes = [' '.join(word for word in theme.split() if word.lower() not in manual_stopwords) for theme in themes]

                # Ensure brand_name is the first theme
                themes.insert(0, brand_name)

                print(f"Brand Name: {brand_name}")
                print("Top Themes:")
                for theme in themes:
                    print(theme)

                return brand_name, themes
            else:
                logging.error("Response object does not contain 'text' attribute.")
                return "", []
        else:
            logging.error("No response from the generative model.")
            return "", []
    except Exception as e:
        logging.error(f"Error in generating themes: {str(e)}")
        return "", []


async def scrape(url: str):
    try:
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")

        # Extract other page content
        scraped_data = await page.evaluate('''() => {
            const content = [];

            const footer = document.querySelector('footer');

            const selectors = ['h1', 'h2', 'h3', 'h4', 'p', 'div', 'span', 'li'];

            // Extract meta title and description
            const metaTitle = document.querySelector('title') ? document.querySelector('title').innerText : '';
            const metaDescription = document.querySelector('meta[name="description"]') ? document.querySelector('meta[name="description"]').getAttribute('content') : '';

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
                'meta_title': metaTitle,
                'meta_description': metaDescription,
                'page_text': content.join(' ')
            };
        }''')

        # Filter page text
        filtered_text = filter_text(scraped_data['page_text'])

        return_data = {
            'full_domain': url,
            'meta_content': '',
            'meta_title': scraped_data['meta_title'],
            'title_tag': scraped_data['meta_title'],
            'page_text': filtered_text
        }

        # Print each field before returning
        print("Full Domain:", return_data['full_domain'])
        print("Meta Title:", return_data['meta_title'])
        print("Title Tag:", return_data['title_tag'])
        print("Page Text:", return_data['page_text'])

        return return_data

    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail="Scraping failed")


def filter_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase and tokenize
    words = text.lower().split()

    # Remove duplicates
    words = list(set(words))

    # Remove stopwords using the manual_stopwords set
    words = [word for word in words if word not in manual_stopwords]

    # Remove strings with only special characters and/or numbers
    words = [word for word in words if re.search(r'[a-zA-Z]', word)]

    return ' '.join(words)

async def generate_ad_copy(brand_name, themes, brand_voice):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        prompt_parts = f"""[
  "System Instructions Information for following prompt:You are a marketing copywriter for a company that describes their writing style like this: {brand_voice}. If there's no specific writing style stated, use a typical marketing voice.",
  "Brand Info You're writing ads using these themes: {themes},
  "Definition of good ad copy: each of the 15 headlines has a unique value proposition, with strong calls-to-actionDefinition of good ad headline: each of the 4 descriptions has a unique value proposition, illustrating the customer benefit of the top themes. Uses product benefits over product features.",
  "input 5: Place the output into a single string separated by a slash. Your response should be a single paragraph with no text other than what's explicitly requested.",
  "Headlines Write 15 ad copy lines that are 7 tokens in length. Capitalize every word.",
  "Descriptions Write 4 ad copy lines that are 20 tokens in length\nUse normal capitalization",
  "output: ",]
        """
        response = await model.generate_content(prompt_parts)
        if response.done:
            # Accessing text content directly from the structured response
            if 'candidates' in response.result and response.result['candidates']:
                content_parts = response.result['candidates'][0]['content']['parts']
                ad_copy_texts = [part['text'] for part in content_parts]
                ad_copy = ' / '.join(ad_copy_texts)  # Joining text parts with slashes as separators
                print("Ad Copy:", ad_copy)
                return ad_copy
            else:
                logging.error("No candidates found in the response.")
                return ""
        else:
            logging.error("Model response is not done.")
            return ""
    except Exception as e:
        logging.error(f"Error in generating ad copy: {str(e)}")
        return ""
    
async def generate_ad_copy_keywords(keywords, brand_voice):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

        prompt_parts = f"""[
  "System Instructions Information for following prompt:You are a marketing copywriter for a company that describes their writing style like this: {brand_voice}. If there's no specific writing style stated, use a typical marketing voice.",
  "Brand Info You're writing ads using these themes: {keywords},
  "Definition of good ad copy: each of the 15 headlines has a unique value proposition, with strong calls-to-actionDefinition of good ad headline: each of the 4 descriptions has a unique value proposition, illustrating the customer benefit of the top themes. Uses product benefits over product features.",
  "input 5: Place the output into a single string separated by a slash. Your response should be a single paragraph with no text other than what's explicitly requested.",
  "Headlines Write 15 ad copy lines that are 7 tokens in length. Capitalize every word.",
  "Descriptions Write 4 ad copy lines that are 20 tokens in length\nUse normal capitalization",
  "output: ",]
        """
        response = await model.generate_content(prompt_parts)
        if response.done:
            # Accessing text content directly from the structured response
            if 'candidates' in response.result and response.result['candidates']:
                content_parts = response.result['candidates'][0]['content']['parts']
                ad_copy_texts = [part['text'] for part in content_parts]
                ad_copy = ' / '.join(ad_copy_texts)  # Joining text parts with slashes as separators
                print("Ad Copy:", ad_copy)
                return ad_copy
            else:
                logging.error("No candidates found in the response.")
                return ""
        else:
            logging.error("Model response is not done.")
            return ""
    except Exception as e:
        logging.error(f"Error in generating ad copy: {str(e)}")
        return ""

@app.post("/analyze", response_model=GenericResponse)
async def analyze(url_or_keywords: str = Form(...), brand_voice: str = Form(None)):
    # Assuming the `generate_ad_copy` or similar function returns the combined list directly
    try:
        if url_or_keywords.startswith(("http://", "https://")):
            scraped_data = await scrape(url_or_keywords)
            brand_name, themes = await generate_themes(scraped_data['full_domain'], scraped_data['meta_content'], scraped_data['meta_title'], scraped_data['title_tag'], scraped_data['page_text'])
            ad_copy = await generate_ad_copy(brand_name, themes, brand_voice)
        else:
            keywords = [keyword.strip() for keyword in url_or_keywords.split(',')]
            ad_copy = await generate_ad_copy_keywords(keywords, brand_voice)
        
        response_content = {
            "meta_title_words": brand_name if url_or_keywords.startswith(("http://", "https://")) else "Keywords Analysis",
            "top_themes_text": themes if url_or_keywords.startswith(("http://", "https://")) else keywords,
            "ad_copy": ad_copy  # This is now the single list containing both headlines and descriptions
        }
        return GenericResponse(content=response_content)

    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in processing")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)