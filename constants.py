sentiment_prompt = """
This is empty for now

"""

complaints_modelling_prompt = """
Identify the complaints or suggestions in the given review.
Each review may have multiple complaints or suggestions also. 
Check if the complaints or suggestions are similar to a complaint or suggestions which is part of this list:{}. 
If yes, return same complaint or suggestions. If not, give the complaints that you have identified. Keep it short and specific. Your response should be in this format. output format : [your list of complaints or suggestions here]. Here is an example output for your reference : ['Unexpected increase in cost', 'Inconsistent quotation'].Please stick to given format only.

Note : If there are no complaints or suggestions identified, return []
"""

positive_points_modelling_prompt = """
Identify the positive points in the given review.
Each review may have multiple positive points also. 
Check if the positive points are similar to a positive point which is part of this list:{}. 
If yes, return same positive points. If not, give the positive points that you have identified. Keep it short and specific. Your response should be in this format. output format : [your list of positive points here]. Here is an example output for your reference : ['Value for money','Excellent customer care service'].Please stick to given format only
"""

change_format_prompt = """
Change the format of given pints to this format.
Output format : ['point1','point2']. 
Here is an example output for your reference : ['Unexpected increase in cost', 'Inconsistent quotation']
Please stick to given format only"
"""

summarize_prompt = """
Accurately summarize all reviews on the given topic, providing a concise and straightforward paragraph that highlights the main content of the reviews regarding the topic given in prompt without using generalizations. This summary will assist the product and business teams in understanding the specific points mentioned in the reviews pertaining to the given topic.
"""

theme_finder_prompt = """
                As a theme-based topic grouper employed by a product review analysis tool, your task is to group topics that discuss similar issues but are phrased differently. These grouped topics should be given a theme name that represents actionable areas for the team to improve the product. By doing so, the team can focus on enhancing the product based on the feedback received.
                
                Your output should be in the following format:

                your_identified_theme_name : ["topic1", "topic2", "topic3"]

                Example:
                Input:
                ["Unexpected Increasing in Price","Demanding Extra Money","Inconsistent quotation","well packed","Good packing","more upi payment options", "show previous completed orders"]

                Example Output:
                {"Pricing Issues": ["Unexpected Increasing in Price","Demanding Extra Money","Inconsistent quotation"],
                 "Packaging": ["well packed","Good packing"],
                 "App suggestions":  : ["more upi payment options", "show previous completed orders"]
                }

                Please group the following topics into appropriate high-level themes without altering the spelling or the case of the individual topics.

                Note: Ensure that you include all the individual topics provided.
"""


