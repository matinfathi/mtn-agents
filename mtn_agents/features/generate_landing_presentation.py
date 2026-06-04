from ..constants import OPENAI_CHAT_MODEL_GPT_4O, JSON_RESPONSE_FORMAT, OPENAI_CHAT_MODEL_GPT_4O_MINI
from ..prompt_templates import prompt_generate_landing_presentation, prompt_generate_landing_presentation_v2
from ..upload_reference.main import upload_presentation_to_index_pinecone
from ..features import auto_generate_slide_concurrent
from ..types import ProductDescriptionDict
from ..utils import Slide, Presentation, hex_uuid
from ..llm import OpenAILLM

from langchain_community.document_loaders import AsyncChromiumLoader
from lxml import etree
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any
from functools import lru_cache
import requests
import logging
import uuid
import json


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_instance() -> OpenAILLM:
    return OpenAILLM()


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_sections(url: str) -> List[str]:
    # TODO make this request once instead of twice
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    sections = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(url, href)

        if url in full_url and "#" in full_url:
            sections.add(full_url)
        elif href.startswith("#"):
            sections.add(url + href)

    return list(sections)


def scrape_section(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    section_id = urlparse(url).fragment

    if section_id:
        section_element = soup.find(id=section_id)
        if section_element:
            return section_element.get_text(strip=True)
        else:
            section_element = soup.find(["section", "div"], class_=section_id)
            if section_element:
                return section_element.get_text(strip=True)

    return None


def get_sections_in_list(base_url: str) -> List[Dict[str, str]]:
    content = []
    try:
        sections = get_sections(base_url)
    except Exception as e:
        raise Exception(f"Could not scrape the website. {e}") from e
    for section in sections:
        scraped_content = scrape_section(section)
        if scraped_content is not None:
            content.append({"url": section, "content": scraped_content})

    return content


def generate_short_uuid() -> str:
    return str(uuid.uuid4())[:8]  # Generate a short UUID (first 8 characters)


def organize_sections(content: List[Dict[str, str]]) -> ProductDescriptionDict:
    llm_instance = get_instance()

    url_mapping = {}
    for item in content:
        short_uuid = hex_uuid()[:8]
        url_mapping[short_uuid] = item["url"]
        item["url"] = short_uuid

    variables = {"content": content}

    assembled_prompt = prompt_generate_landing_presentation.text.format(**variables)
    raw_response = llm_instance.query(
        prompt=assembled_prompt, model=OPENAI_CHAT_MODEL_GPT_4O, response_format=JSON_RESPONSE_FORMAT
    )

    result = json.loads(raw_response)

    slides = []
    for idx, item in enumerate(result["slides"]):
        original_url = url_mapping.get(item["url"], item["url"])
        slides.append(Slide(slide_number=f"page_{idx+1}", slide_content=item["content"], url=original_url))

    return dict(Presentation(slides))


def remove_tags(
    doc: Any, tags: List[str] = ["script", "style", "meta", "link", "svg", "path", "button", "img", "head"]
) -> BeautifulSoup:
    html_content = doc.page_content
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in tags:
        for element in soup.find_all(tag):
            element.decompose()
    return soup


def tree_maker(soup: BeautifulSoup) -> etree._Element:
    return etree.fromstring(str(soup), etree.HTMLParser())


def get_full_xpath(element: etree._Element) -> str:
    tree = element.getroottree()
    return tree.getpath(element)


def build_automata(element: etree._Element, parent_full_path: str = None, automata: Dict = None) -> Dict:
    if automata is None:
        automata = {}

    full_path = get_full_xpath(element)
    relative_path = full_path if parent_full_path is None else full_path.replace(parent_full_path, "").lstrip("/")
    text = (element.text or "").strip()
    tail = (element.tail or "").strip()
    combined_text = text + tail
    children = list(element)
    while len(children) == 1 and not combined_text:
        child = children[0]
        full_path = get_full_xpath(child)
        relative_path = f"{relative_path}/{full_path.replace(get_full_xpath(element), '').lstrip('/')}"
        element = child
        combined_text = (element.text or "").strip() + (element.tail or "").strip()
        children = list(element)
    if combined_text or children:
        data = {"text": combined_text} if combined_text else {}
        children_data = {}
        for child in children:
            child_automata = build_automata(child, get_full_xpath(element), {})
            if child_automata:
                children_data.update(child_automata)
        if children_data:
            data.update(children_data)
        automata[relative_path] = data

    return automata


def get_page_content(url: str) -> Any:
    try:
        loader = AsyncChromiumLoader([url])
        docs = loader.load()
    except Exception as e:
        raise Exception(f"Loading url {url} failed. Error {e}")

    return docs[0]


def _get_sections(automata: Dict) -> Dict:
    llm_instance = get_instance()
    variables = {"content": automata}
    prompt = prompt_generate_landing_presentation_v2.text.format(**variables)

    result = llm_instance.reasoning(
        prompt=prompt,
        model="o4-mini",
        # response_format={"type": "json_object"},
    )

    result = result.strip()
    if result.startswith('```'):
        result = result.split('\n', 1)[1]
        result = '\n'.join(result.split('\n')[:-1])

    # print(result)

    return json.loads(result)["sections"]


def get_sections_v2(url: List[Dict[str, str]]) -> Dict:
    page_content = get_page_content(url)
    cleaned_content = remove_tags(page_content)
    tree = tree_maker(cleaned_content)
    automata = build_automata(tree)
    sections = _get_sections(automata)

    for section in sections:
        sections_temp = section.pop("path_sections")
        url = section.pop("path_main")
        if not sections_temp:
            raise ValueError("One of the site sections has no Xpath.")
        section["url"] = url

    return sections


def auto_generate_landing_presentation(
    url: str, agent_uuid: str, user_id: str, run_auto_slide=False, document_ids: List[str] = None
) -> ProductDescriptionDict:
    sections_pd = get_sections_v2(url)

    # sections_pd = get_sections(url)
    # sections_pd = organize_sections(sections)

    present = dict(
        Presentation(
            [
                Slide(slide_number=f"page_{idx+1}", slide_content=item["text_section"], url=item["url"])
                for idx, item in enumerate(sections_pd)
            ]
        )
    )

    if not sections_pd:
        raise Exception("No sections found in the website.")

    if run_auto_slide:
        final_presentation = auto_generate_slide_concurrent(
            slide_path=present, user_id=user_id, document_ids=document_ids
        )
    else:
        final_presentation = present

    uploaded_data = upload_presentation_to_index_pinecone(
        presentation=final_presentation, user_id=user_id, agent_uuid=agent_uuid
    )

    logger.info(f"\n----------------------\nPresentation in pinecone: {uploaded_data}\n----------------------\n")

    return final_presentation
