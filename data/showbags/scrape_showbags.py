# prior to running this script, ensure playright is setup
# this was executed in a wsl environment using chromiu
# python -m playwright install chromium
# python -m playwright install-deps

import time
import os
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import json

def extract_showbags_from_html(html_content, base_url="https://www.eastershow.com.au"):
    """Extract showbag information from HTML content using BeautifulSoup"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all showbag cards
    showbag_cards = soup.find_all('div', class_='showbagsCard')
    
    showbags = []
    
    for card in showbag_cards:
        # Extract showbag ID
        showbag_id = card.get('data-id', '')
        
        # Extract showbag name
        name_element = card.find('h3', class_='showbagsCard-product--name')
        name = name_element.text.strip() if name_element else "Unknown"
        
        # Extract price
        price_element = card.find('span', class_='showbagsCard-product--price')
        price = price_element.text.strip() if price_element else "Unknown"
        
        # Extract stand numbers
        stand_info_element = card.find('span', class_='showbagsCard-product--info')
        stand_numbers = stand_info_element.text.strip() if stand_info_element else "Unknown"
        
        # Extract image URL
        image_url = ""
        img_element = card.select_one('.showbagsCard-image img')
        if img_element and img_element.has_attr('src'):
            image_url = img_element['src']
            # Make sure URL is absolute
            if image_url.startswith('/'):
                image_url = base_url + image_url
        
        # Extract included items and retail value
        included_items = []
        retail_value = ""
        
        description_section = card.find('div', class_='showbagsCard-description')
        if description_section:
            included_section = description_section.find('div', class_='showbagsCard-description-copy--included')
            if included_section:
                items = included_section.find_all('p')
                for item in items:
                    item_text = item.text.strip()
                    if "Total Retail Value:" in item_text:
                        retail_value = item_text
                    else:
                        included_items.append(item_text)
        
        # Extract distributor
        distributor = ""
        if description_section:
            distributor_section = description_section.find('div', class_='showbagsCard-description-copy--distributor')
            if distributor_section:
                distributor_text = distributor_section.text.strip()
                distributor = distributor_text.split('<br>')[0] if '<br>' in distributor_text else distributor_text
        
        showbag_data = {
            'id': showbag_id,
            'name': name,
            'price': price,
            'stand_numbers': stand_numbers,
            'image_url': image_url,
            'included_items': included_items,
            'retail_value': retail_value,
            'distributor': distributor
        }
        
        showbags.append(showbag_data)
    
    return showbags

def save_page_source(html, page_num, directory="html_pages"):
    """Save the page source to a file for debugging"""
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = os.path.join(directory, f"page_{page_num}.html")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Saved page source to {filename}")

def scrape_with_playwright(max_pages=38, save_html=True, debug=True, headless=True):
    """Use Playwright to scrape showbags data by navigating through pagination"""
    all_showbags = []
    base_url = "https://www.eastershow.com.au"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = context.new_page()
        
        # Enable request interception for debugging if needed
        if debug:
            page.on("request", lambda request: print(f">> {request.method} {request.url}") if "ShowBagCategoryPage" in request.url else None)
            page.on("response", lambda response: print(f"<< {response.status} {response.url}") if "ShowBagCategoryPage" in response.url else None)
        
        try:
            # Navigate to the showbags page
            url = f"{base_url}/explore/showbags/"
            print(f"Navigating to {url}")
            page.goto(url, wait_until="networkidle")
            
            # Wait for the page to load
            page.wait_for_selector(".showbagsCard", timeout=10000)
            
            # Extract data from the first page
            print("Extracting data from page 1")
            page1_html = page.content()
            
            if save_html:
                save_page_source(page1_html, 1)
            
            page1_showbags = extract_showbags_from_html(page1_html, base_url)
            all_showbags.extend(page1_showbags)
            print(f"Extracted {len(page1_showbags)} showbags from page 1")
            
            # Process remaining pages
            current_page = 1
            
            while current_page < max_pages:
                try:
                    # Find the next page button
                    next_button_selector = ".pagination-list-btnnav-next"
                    
                    # Check if next button exists
                    if not page.query_selector(next_button_selector):
                        print("Next button not found, stopping pagination")
                        break
                    
                    # Get the next page number from the button
                    next_page_num = int(page.evaluate(f"document.querySelector('{next_button_selector}').getAttribute('data-pageno')"))
                    print(f"Found next page button pointing to page {next_page_num}")
                    
                    # Click the next page button
                    print(f"Clicking the button to navigate to page {next_page_num}")
                    page.click(next_button_selector)
                    
                    # Wait for the pagination to update
                    page.wait_for_selector(f".pagination-list-btnpage[data-pageno='{next_page_num}'].is-current", timeout=10000)
                    page.wait_for_load_state("networkidle")
                    
                    # Ensure the page has loaded properly by waiting for showbag cards
                    page.wait_for_selector(".showbagsCard", timeout=10000)
                    
                    # Update current page number
                    current_page = next_page_num
                    print(f"Successfully navigated to page {current_page}")
                    
                    # Save the page source for debugging if requested
                    if save_html:
                        page_html = page.content()
                        save_page_source(page_html, current_page)
                    
                    # Extract showbag data from the current page
                    page_showbags = extract_showbags_from_html(page.content(), base_url)
                    
                    if page_showbags:
                        all_showbags.extend(page_showbags)
                        print(f"Extracted {len(page_showbags)} showbags from page {current_page}")
                    else:
                        print(f"No showbags found on page {current_page}, stopping pagination")
                        break
                    
                    # Pause to avoid overloading the server
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing page {current_page + 1}: {str(e)}")
                    if debug:
                        # Take screenshot for debugging
                        page.screenshot(path=f"error_page_{current_page + 1}.png")
                    break
            
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            if debug:
                page.screenshot(path="error_screenshot.png")
        
        finally:
            # Always close the browser
            browser.close()
    
    return all_showbags

def process_and_save_data(showbags, filename="showbags.csv"):
    """Process the extracted data and save to CSV and JSON files"""
    if not showbags:
        print("No showbags data to process")
        return
    
    # Create DataFrame
    df = pd.DataFrame(showbags)
    
    # Process price column - remove $ and convert to numeric
    if 'price' in df.columns:
        df['price_numeric'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Extract numeric retail value
    if 'retail_value' in df.columns:
        df['retail_value_numeric'] = df['retail_value'].str.extract(r'Total Retail Value: \$([0-9\.]+)').astype(float)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to CSV: {filename}")
    
    # Also save as JSON for easier programmatic access
    json_filename = filename.replace('.csv', '.json')
    df.to_json(json_filename, orient='records')
    print(f"Data saved to JSON: {json_filename}")
    
    return df

def main():
    print("Starting the Sydney Royal Easter Show showbags scraping process...")
    
    # Scrape the data - set headless=False if you want to see the browser
    showbags = scrape_with_playwright(max_pages=38, save_html=True, debug=True, headless=True)
    
    # Process and save the data
    df = process_and_save_data(showbags)
    
    # Display summary
    print(f"Scraping complete. Collected {len(df) if df is not None else 0} showbags in total.")
    
    # Display a sample
    if df is not None and not df.empty:
        print("\nSample of scraped data:")
        sample_columns = ['name', 'price', 'image_url']
        print(df[sample_columns].head())

if __name__ == "__main__":
    main()