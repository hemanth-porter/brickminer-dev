import functools
import time
import openai
import logging

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        return result
    return wrapper


def retry(func):
        """
        
        retries executing the same function if it falils for 5 times after waiting for a second. 

        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            count = 0

            while count < 5:
                try:
                    result = func(*args, **kwargs)
                    return result    

                except openai.error.RateLimitError as e:
                    logging.info("Rate Limit Hit, Sleeping for 60 Seconds")
                    count += 1
                    time.sleep(60)
                                        
                except Exception as e:
                    logging.error(f"{func.__name__} failed with error: {str(e)}")
                    count += 1
                    time.sleep(2)
            
            logging.error("GPT error after 5 attempts")
            
            return "Error after 5 attempts"
        
        return wrapper