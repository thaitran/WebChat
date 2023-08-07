import inspect
import re
import urllib.parse
from bs4 import BeautifulSoup, Comment
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from math import *

_tools = { }   # Dict of available tools
_webdriver = None   # Selenium webdriver for Chrome browser

def add_tool(func):
    """
    Adds a Python function as an available tool for the LLM.  This is intended
    to be used as a decorator on the function definition.  For example:

    @add_tool
    def google_search(topic):
        ...

    The tool name is converted from the Python convention to UpperCaseCamelCase.
    For example: google_search => GoogleSearch

    The tool description is taken from the function's docstring.
    """
    global _tools

    def transform_tool_name(name):
        words = name.split("_")
        return "".join(word.capitalize() for word in words)

    tool_name = transform_tool_name(func.__name__)
    tool_desc = func.__doc__.strip()

    params = inspect.signature(func).parameters
    tool_params = list(params.keys())

    _tools[tool_name] = {
        "params": tool_params,
        "desc": tool_desc,
        "func": func
    }

    return func

def tools_prompt():
    """
    Returns a string of all the available tools for inclusion in the system
    prompt sent to the LLM.  For example:

    * Calculate[ expression ] - Evaluate a mathematical expression...
    * GoogleSearch [ topic ] - Search Google for a topic...
    """
    global _tools

    tools_prompt = ""

    for name, tool in _tools.items():
        param_str = ",".join(tool["params"])
        tools_prompt += f"""* {name}[ {param_str} ] - {tool["desc"]}\n"""

    return tools_prompt

def run_tool(name, params):
    """
    Runs a tool and returns the result.
    """
    global _tools

    if not name in _tools:
        return f"{name}[] is not a valid tool"
    
    # The LLM sometimes puts double quotes around the param
    params = params.strip('"')

    result = _tools[name]["func"](params)

    return result

def get_webdriver():
    """
    Get the Selenium webdriver for a Chrome browser.
    This will create the browser if it's doesn't already exist.
    """
    global _webdriver

    options = Options()
    options.headless = False

    if _webdriver is None:
        _webdriver = webdriver.Chrome(options=options)
        return _webdriver
    
    # Check if the web browser is still alive.  If not, start a new one.
    try:
        title = _webdriver.title
    except:
        _webdriver = webdriver.Chrome(options=options)

    return _webdriver

def distill_html(raw_html, remove_links=False):
    """
    Reduce HTML to the minimal tags necessary to understand the content.
    Set remove_links=True to also replace <a> tags with their inner content.
    """
    soup = BeautifulSoup(raw_html, 'html.parser')

    # Tags (with inner content) that should be completely removed from the HTML
    # Note:  We want to keep <g-section-with-header> as it shows Top Stories
    remove_tags = [
        'aside', 'br', 'button', 'cite', 'cnx', 'fieldset', 'figcaption', 
        'figure', 'footer', 'form', 'g-dropdown-button', 
        'g-dropdown-menu-button', 'g-fab', 'g-img', 'g-inner-card', 
        'g-left-button', 'g-link', 'g-loading-icon', 'g-more-linkg-menu-item', 
        'g-popup', 'g-radio-button-group', 'g-right-button', 
        'g-scrolling-carousel', 'g-snackbar', 'g-white-loading-icon',
        'google-read-aloud-player', 'head', 'hr', 'iframe', 'img', 'input', 
        'label', 'link', 'meta', 'nav', 'next-route-announcer', 'noscript', 
        'option', 'promo-throttler', 'script', 'select', 'style', 'svg'
    ]
    valid_attrs = ['href']

    # Remove all unwanted tags
    for script in soup(remove_tags):
        script.decompose()

    # Remove all unwanted attributes
    for tag in soup():
        attrs = dict(tag.attrs)
        for attr in attrs:
            if attr not in valid_attrs:
                del tag[attr]

    # Replace every <span> and <p> with it's inner contents
    for span in soup.find_all(['span', 'p']):
        span.replace_with(" " + span.text + " ")

    # Replace links with plain text
    if remove_links:
        for link in soup.find_all('a'):
            link.replace_with(" " + link.text + " ")
            
    # Remove comments
    for comment in soup.findAll(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove empty divs (e.g. <div> </div>)
    for div in soup.find_all("div"):
        if (div.text is None) or (div.text.strip() == ""):
            div.decompose()

    # Compress nested divs.  For example:
    # <div><div><div>Content</div></div></div> -> <div>Content>/div>)
    for div in soup.find_all("div"):
        children = div.findChildren(recursive=False)
        if len(children) == 1 and children[0].name == 'div':
            div.replace_with(children[0])

    html = str(soup)

    # Compress whitespace
    html = re.sub(r'(\s|\n)+', ' ', html)

    return html

def safe_eval(expression):
    """
    A version of eval() that only allows a limited set of math functions.
    """

    safe_list = [
        'abs', 'acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 
        'degrees', 'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 'hypot', 
        'ldexp', 'log', 'log10', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 
        'sqrt', 'tan', 'tanh'
    ]

    safe_dict = dict([ (k, locals().get(k, None)) for k in safe_list ])

    return eval(expression, { "__builtins__": None }, safe_dict)


@add_tool
def calculate(expression):
    """Evaluate a mathematical expression using Python.  Expression should only contain numbers, operators (+ - * / **), or math module functions."""
    
    expression = expression.replace("^", "**")  # The LLM sometimes uses ^ for exponentation. Replace this operator with **

    try:
        return str(safe_eval(expression))
    except:
        return "That was not a valid expression"

@add_tool
def google_search(topic):
    """Use Google to search the web for the topic."""

    driver = get_webdriver()

    driver.get("https://www.google.com/search?q=" + urllib.parse.quote(topic))
    html = distill_html(driver.page_source)

    soup = BeautifulSoup(html, 'html.parser')

    # Remove internal Google links
    strip_links = [
        "/", 
        "#", 
        "https://www.google.com", 
        "https://maps.google.com", 
        "https://support.google.com",
        "https://policies.google.com",
        "https://accounts.google.com"
    ]
    for link in soup.find_all('a'):
        url = link.get('href')
        if url:
            for remove_link in strip_links:
                if url.startswith(remove_link):
                    link.decompose()
                    break

    html = str(soup)

    # Remove everything before <h1>Search Results</h1>
    match = re.search(r"<h1>Search Results<\/h1>", html)
    if match:
        start_position = match.start()
        html = html[start_position:]

    # Remove everything after <h1>Page Navigation</h1>
    match = re.search(r"<h1>Page Navigation<\/h1>", html)
    if match:
        start_position = match.start()
        html = html[:start_position]

    return html

@add_tool
def get_web_page(url):
    """Get the contents of a web page. Only call this with a valid URL."""

    driver = get_webdriver()

    try:
        driver.get(url)
        return distill_html(driver.page_source, remove_links=True)
    except:
        return "Error retrieving web page"

