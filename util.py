import re
import math
from bs4 import BeautifulSoup, Comment

def safe_eval(expression):
    """
    A version of eval() that only allows a limited set of math functions.
    """

    safe_list = [
        'acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh',
        'degrees', 'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 'hypot',
        'ldexp', 'log', 'log10', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh',
        'sqrt', 'tan', 'tanh'
    ]

    safe_dict = {k: getattr(math, k) for k in safe_list}
    safe_dict['abs'] = abs

    try:
        return eval(expression, {"__builtins__": None}, safe_dict)
    except Exception as e:
        raise ValueError(f'Error evaluating expression: {e}')

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
        'label', 'link', 'nav', 'next-route-announcer', 'noscript', 
        'option', 'promo-throttler', 'script', 'select', 'style', 'svg'
    ]
    valid_attrs = ['href']

    # Remove all unwanted tags
    for tag in soup(remove_tags):
        tag.decompose()

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
