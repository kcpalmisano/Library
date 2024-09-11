# -*- coding: utf-8 -*-
"""
@author: cpalmisano

This is a huge work in progress... 
Doing this for funsies and not for funsies 

New to this type of website based interaction so progress is slow. 

"""

import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import logging
import time
import os
from jinja2 import Template

# Set up logging
logging.basicConfig(level=logging.INFO, filename='C:/Users/path/job_bot.log', format='%(asctime)s:%(levelname)s:%(message)s')

###############  LINKEDIN 
# def linkedin_login(driver, username, password):
#     logging.info("Logging into LinkedIn")
    
#     # Load LinkedIn login page
#     driver.get("https://www.linkedin.com/login")
    
#     # Wait for the login page to load completely
#     time.sleep(5)
    
#     # Find and fill the username and password fields
#     username_field = driver.find_element(By.ID, "username")
#     password_field = driver.find_element(By.ID, "password")
    
#     username_field.send_keys(username)
#     password_field.send_keys(password)
    
#     # Submit the login form
#     password_field.send_keys(Keys.RETURN)
    
#     # Wait for the login process to complete
#     time.sleep(5)
    
#     logging.info("Logged in successfully")


# def apply_to_job(driver):
#     try:
#         logging.info("Attempting to apply for job...")
#         # Wait for the "Easy Apply" button to appear and click it
#         easy_apply_button = WebDriverWait(driver, 10).until(
#             EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'Easy Apply')]"))
#         )
#         easy_apply_button.click()
        
#         time.sleep(3)
        
#         # Fill in the application form
#         submit_button = WebDriverWait(driver, 10).until(
#             EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'Submit application')]"))
#         )
#         submit_button.click()

#         logging.info("Successfully applied to the job!")
#     except TimeoutException:
#         logging.warning("Failed to apply to the job (No Easy Apply button).")
#     except NoSuchElementException:
#         logging.warning("Failed to locate application elements.")
#     except Exception as e:
#         logging.error(f"An error occurred while trying to apply: {e}")

# def scrape_and_apply_jobs(keywords, url, username, password):
#     logging.info(f"Starting job application bot for keywords: {keywords}")
    
#     driver = webdriver.Chrome()
#     linkedin_login(driver, username, password)
#     driver.get(url)
    
#     try:
#         logging.info("Waiting for the search input to be present and interactable...")
#         search_input = WebDriverWait(driver, 20).until(
#             EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Search by title, skill, or company']"))
#         )
#         search_input.send_keys(keywords)
#         search_input.send_keys(Keys.RETURN)
        
#         logging.info("Waiting for job listings to load...")
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.XPATH, "//ul[contains(@class, 'jobs-search__results-list')]/li"))
#         )
#         logging.info("Job listings loaded.")
        
#         # Click on the first job listing and apply
#         job_listings = driver.find_elements(By.XPATH, "//ul[contains(@class, 'jobs-search__results-list')]/li")
        
#         for job in job_listings:
#             job_title_element = job.find_element(By.XPATH, ".//a[contains(@class, 'job-card-list__title')]")
#             job_title = job_title_element.text.strip()
            
#             if keywords.lower() in job_title.lower():
#                 logging.info(f"Applying for job: {job_title}")
#                 job_title_element.click()
#                 time.sleep(3)  # Wait for the job details page to load
                
#                 # Apply to the job
#                 apply_to_job(driver)
                
#                 time.sleep(3)  # Small delay before moving to the next job listing
#                 driver.back(2)  # Go back to the job listings page
#                 time.sleep(3)  # Wait for the page to load again
                
#     except TimeoutException:
#         logging.error("Timeout waiting for job listings to load or interact.")
#     finally:
#         driver.quit()

# Example usage
email = 'email@gmail.com'
password = 'PassWord!'

###############  INDEED 

def indeed_google_login(driver, email, password):
    logging.info("Logging into Indeed with Google")
    
    driver.get("https://secure.indeed.com/account/login")
    
    try:
        # Wait for the "Continue with Google" button and click it
        google_sign_in_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Continue with Google']"))
        )
        logging.info("Found 'Continue with Google' button, clicking it.")
        google_sign_in_button.click()
        
        # Wait for a new window to open and switch to it
        WebDriverWait(driver, 20).until(EC.number_of_windows_to_be(2))
        window_handles = driver.window_handles
        logging.info(f"Window handles after clicking Google Sign-in: {window_handles}")
        
        # Switch to the new Google login window
        driver.switch_to.window(window_handles[-1])
        logging.info(f"Switched to Google login window. Current page title: {driver.title}")
        
        # Wait for the Google email input field using a broader XPath (fallback)
        email_field = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//input[@type='email']"))
        )
        logging.info("Email input field located. Entering email.")
        email_field.send_keys(email)
        email_field.send_keys(Keys.RETURN)

        # Wait for the password input field using a broader XPath
        password_field = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//input[@type='password']"))
        )
        
        # Scroll into view if not visible
        driver.execute_script("arguments[0].scrollIntoView(true);", password_field)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(password_field))
        
        logging.info("Password input field visible and interactable. Entering password.")
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)
        
        logging.info("Google login credentials submitted.")
        
        # Switch back to the original Indeed window
        WebDriverWait(driver, 15).until(EC.number_of_windows_to_be(1))
        driver.switch_to.window(window_handles[0])
        logging.info(f"Switched back to Indeed window. Current page title: {driver.title}")
        
    except TimeoutException as e:
        logging.error(f"Timeout occurred during Google login: {e}")
        logging.error(f"Current page title: {driver.title if driver.title else 'Unknown'}")
        logging.error(f"Window handles at timeout: {driver.window_handles}")
        screenshot_path = os.path.join(os.getcwd(), 'google_login_error.png')
        driver.save_screenshot(screenshot_path)
        logging.info(f"Screenshot saved to {screenshot_path}")
    except NoSuchElementException as e:
        logging.error(f"Element not found during login: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")



def scrape_and_apply_jobs(keywords, location, email, password):
    logging.info(f"Starting job application bot for keywords: {keywords}")
    
    driver = webdriver.Chrome()
    indeed_google_login(driver, email, password)
    
    driver.get("https://www.indeed.com/")
    
    try:
        # Clear and set the job title and location using JavaScript
        search_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "text-input-what"))
        )
        location_input = driver.find_element(By.ID, "text-input-where")
        
        # Clear the fields using JavaScript
        driver.execute_script("arguments[0].value = '';", search_input)
        driver.execute_script("arguments[0].value = '';", location_input)
        
        # Set the fields using JavaScript
        driver.execute_script(f"arguments[0].value = '{keywords}';", search_input)
        driver.execute_script(f"arguments[0].value = '{location}';", location_input)
        
        # Verify the values have been correctly set
        assert search_input.get_attribute("value") == keywords, "Job title was not correctly set"
        assert location_input.get_attribute("value").strip().lower() == location.strip().lower(), "Location was not correctly set"
        
        # Start the search by pressing ENTER on the search input field
        search_input.send_keys(Keys.RETURN)
        
        logging.info("Waiting for job listings to load...")
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'jobsearch-SerpJobCard')]"))
        )
        logging.info("Job listings loaded.")
        
        job_listings = driver.find_elements(By.XPATH, "//div[contains(@class, 'jobsearch-SerpJobCard')]")
        
        for job in job_listings:
            try:
                job_title_element = job.find_element(By.XPATH, ".//h2[contains(@class, 'title')]")
                job_title = job_title_element.text.strip()
                
                if keywords.lower() in job_title.lower():
                    logging.info(f"Applying for job: {job_title}")
                    job_title_element.click()
                    time.sleep(3)  # Wait for the job details page to load
                    
                    apply_to_job(driver)
                    
                    time.sleep(2)
                    driver.back()
                    time.sleep(2)
            except Exception as e:
                logging.warning(f"Failed to click on job or apply: {e}")
                
    except TimeoutException:
        logging.error("Timeout waiting for job listings to load or interact.")
    except AssertionError as e:
        logging.error(f"Assertion error: {e}")
    finally:
        driver.quit()

def apply_to_job(driver):
    try:
        logging.info("Attempting to apply for job...")
        
        apply_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'Apply Now')] | //button[contains(@aria-label, 'Easy Apply')]"))
        )
        apply_button.click()
        
        time.sleep(2)
        
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@type, 'submit')]"))
        )
        submit_button.click()

        logging.info("Successfully applied to the job!")
    except TimeoutException:
        logging.warning("Failed to apply to the job (No Apply Now/Easy Apply button).")
    except NoSuchElementException:
        logging.warning("Failed to locate application elements.")
    except Exception as e:
        logging.error(f"An error occurred while trying to apply: {e}")
        
email = 'email@gmail.com'
password = 'PassWord!'

scrape_and_apply_jobs('Data Engineer', " Remote", email, password)







# indeed_url = "https://www.indeed.com/jobs?q=data+engineer+remote&l=New+York%2C+NY&from=searchOnDesktopSerp&vjk=bd43a90a92c5ad5f"

# scrape_and_apply_jobs('Data Engineer', indeed_url , username, password)
















# def match_keywords(job_description, keywords):
#     # Basic keyword matching logic
#     matches = [kw for kw in keywords if kw in job_description]
#     return len(matches)

# def generate_resume_and_cover_letter(job_details, base_resume, base_cover_letter):
#     # Modify resume and cover letter templates
#     resume_template = Template(base_resume)
#     cover_letter_template = Template(base_cover_letter)
    
#     resume = resume_template.render(job_details)
#     cover_letter = cover_letter_template.render(job_details)
    
#     return resume, cover_letter

# def submit_application(job_url, resume, cover_letter):
#     driver = webdriver.Chrome()  # Or another browser driver
#     driver.get(job_url)
#     # Automate the submission process using Selenium
#     driver.quit()

# def handle_captcha():
#     # Implement captcha handling logic
#     pass

# def main():
#     keywords = ['Python', 'Data Science', 'Machine Learning']
#     job_sites = ['https://example.com/jobs', 'https://anotherexample.com/jobs']
    
#     base_resume = "Path to base resume template"
#     base_cover_letter = "Path to base cover letter template"
    
#     for site in job_sites:
#         job_listings = scrape_job_listings(keywords, site)
#         for job in job_listings:
#             if match_keywords(job['description'], keywords) > 0:
#                 resume, cover_letter = generate_resume_and_cover_letter(job, base_resume, base_cover_letter)
#                 submit_application(job['url'], resume, cover_letter)

# if __name__ == "__main__":
#     main()
