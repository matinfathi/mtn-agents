from mtn_agents import prompt_archive
from collections import namedtuple

Prompt = namedtuple("Prompt", ["text", "variables"])

# AGENTS ---------------------------------------------------------------------------------------------------------------
prompt_master_chain = Prompt(
    text=prompt_archive.text_prompt_master_chain,
    variables=prompt_archive.variables_prompt_master_chain,
)

prompt_customer_support_agent = Prompt(
    text=prompt_archive.text_prompt_customer_support_agent,
    variables=prompt_archive.variables_prompt_customer_support_agent,
)

prompt_video_agent = Prompt(
    text=prompt_archive.text_prompt_video_agent,
    variables=prompt_archive.variables_prompt_video_agent,
)

prompt_voice_sales_agent = Prompt(
    text=prompt_archive.text_prompt_voice_sales_agent,
    variables=prompt_archive.variables_prompt_voice_sales_agent,
)

prompt_visual_sales_agent_v1 = Prompt(
    text=prompt_archive.text_prompt_visual_sales_agent_v1,
    variables=prompt_archive.variables_prompt_visual_sales_agent_v1,
)

prompt_visual_sales_agent_v2 = Prompt(
    text=prompt_archive.text_prompt_visual_sales_agent_v2,
    variables=prompt_archive.variables_prompt_visual_sales_agent_v2,
)

prompt_visual_sales_agent_v3 = Prompt(
    text=prompt_archive.text_prompt_visual_sales_agent_v3,
    variables=prompt_archive.variables_prompt_visual_sales_agent_v3,
)

prompt_visual_sales_agent_v35 = Prompt(
    text=prompt_archive.text_prompt_visual_sales_agent_v35,
    variables=prompt_archive.variables_prompt_visual_sales_agent_v35,
)

prompt_visual_sales_agent_v36 = Prompt(
    text=prompt_archive.text_prompt_visual_sales_agent_v36,
    variables=prompt_archive.variables_prompt_visual_sales_agent_v36,
)

prompt_visual_sales_agent_v4 = Prompt(
    text=prompt_archive.text_prompt_visual_sales_agent_v4,
    variables=prompt_archive.variables_prompt_visual_sales_agent_v4,
)

prompt_master_chain_visual = Prompt(
    text=prompt_archive.text_prompt_master_chain_visual,
    variables=prompt_archive.variables_prompt_master_chain_visual,
)

prompt_teach_agent = Prompt(
    text=prompt_archive.text_prompt_teach_agent,
    variables=prompt_archive.variables_prompt_teach_agent,
)

prompt_site_sage_agent = Prompt(
    text=prompt_archive.text_prompt_site_sage_agent,
    variables=prompt_archive.variables_prompt_site_sage_agent,
)

# ----------------------------------------------------------------------------------------------------------------------
# FEATURES -------------------------------------------------------------------------------------------------------------
prompt_suggested_question = Prompt(
    text=prompt_archive.text_prompt_suggested_question,
    variables=prompt_archive.variables_prompt_suggested_question,
)

prompt_suggested_question_v2 = Prompt(
    text=prompt_archive.text_prompt_suggested_question_v2,
    variables=prompt_archive.variables_prompt_suggested_question_v2,
)

prompt_conversation_insights = Prompt(
    text=prompt_archive.text_prompt_conversation_insights,
    variables=prompt_archive.variables_prompt_conversation_insights,
)

prompt_generate_slide_bullet_points = Prompt(
    text=prompt_archive.text_prompt_generate_slide_bullet_points,
    variables=prompt_archive.variables_prompt_generate_slide_bullet_points,
)

prompt_generate_slide_description = Prompt(
    text=prompt_archive.text_prompt_generate_slide_description,
    variables=prompt_archive.variables_prompt_generate_slide_description,
)

prompt_format_slide_texts = Prompt(
    text=prompt_archive.text_prompt_format_slide_texts,
    variables=prompt_archive.variables_prompt_format_slide_texts,
)

prompt_most_common_keywords = Prompt(
    text=prompt_archive.text_prompt_most_common_keywords,
    variables=prompt_archive.variables_prompt_most_common_keywords,
)

prompt_most_common_keywords_v2 = Prompt(
    text=prompt_archive.text_prompt_most_common_keywords_v2,
    variables=prompt_archive.variables_prompt_most_common_keywords_v2,
)

prompt_slide_feedback = Prompt(
    text=prompt_archive.text_prompt_slide_feedback,
    variables=prompt_archive.variables_prompt_slide_feedback,
)

prompt_generate_landing_presentation = Prompt(
    text=prompt_archive.text_prompt_generate_landing_presentation,
    variables=prompt_archive.variables_prompt_generate_landing_presentation
)


prompt_generate_landing_presentation_v2 = Prompt(
    text=prompt_archive.text_prompt_generate_landing_presentation_v2,
    variables=prompt_archive.variables_prompt_generate_landing_presentation_v2
)

prompt_generate_landing_presentation_v3 = Prompt(
    text=prompt_archive.text_prompt_generate_landing_presentation_v3,
    variables=prompt_archive.variables_prompt_generate_landing_presentation_v3
)
# ----------------------------------------------------------------------------------------------------------------------
# OTHER PROMPTS --------------------------------------------------------------------------------------------------------
prompt_standalone_question = Prompt(
    text=prompt_archive.text_prompt_standalone_question,
    variables=prompt_archive.variables_prompt_standalone_question,
)

prompt_chitchat = Prompt(
    text=prompt_archive.text_prompt_chitchat,
    variables=prompt_archive.variables_prompt_chitchat,
)

prompt_call_to_action = ...

# ----------------------------------------------------------------------------------------------------------------------
