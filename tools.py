import inspect
import re
import urllib.parse
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from util import safe_eval, distill_html

class Tools:
    """
    Tools that can be used by an LLM
    """
    SUPPORTED_BROWSERS = ["Chrome", "Firefox", "Safari", "Edge", "Headless"]

    def __init__(self, browser=None):
        self.tools = { }
        self.webdriver = None

        self.set_browser(browser)

        self.add_tool(
            self.calculate,
            "Calculate",
            "Evaluate a mathematical expression using Python.  Expression should only contain numbers, operators (+ - * / **), or math module functions.")

        self.add_tool(
            self.google_search,
            "GoogleSearch",
            "Use Google to search the web for the topic.")

        self.add_tool(
            self.get_web_page,
            "GetWebPage",
            "Get the contents of a web page. Only call this with a valid URL.")

    def add_tool(self, func, name, desc):
        """
        Adds a Python function as an available tool for the LLM.
        The tool name and desc will be included in the LLM system message.
        """
        params = inspect.signature(func).parameters
        tool_params = list(params.keys())

        self.tools[name] = {
            "params": tool_params,
            "desc": desc,
            "func": func
        }

    def get_tool_list_for_prompt(self):
        """
        Returns a string of all the available tools for inclusion in the system
        prompt sent to the LLM.  For example:

        * Calculate[ expression ] - Evaluate a mathematical expression...
        * GoogleSearch [ topic ] - Search Google for a topic...
        """
        tools_prompt = ""

        for name, tool in self.tools.items():
            param_str = ",".join(tool["params"])
            tools_prompt += f"""* {name}[ {param_str} ] - {tool["desc"]}\n"""

        return tools_prompt

    def run_tool(self, name, params):
        """
        Runs a tool and returns the result.
        """
        if not name in self.tools:
            return f"{name}[] is not a valid tool"
    
        tool = self.tools[name]

        # The LLM sometimes puts double quotes around the param
        params = params.strip('"')

        result = tool["func"](params)

        return result

    def set_browser(self, browser):
        """
        Set the browser that should be used for fetching web pages
        """
        if not(browser is None or browser in self.SUPPORTED_BROWSERS):
            raise Exception(f"The only supported browsers are: {self.SUPPORTED_BROWSERS}")

        self.browser = browser

        # Close the previous browser
        try:
            if self.webdriver:
                self.webdriver.quit()
        except:
            pass

        self.webdriver = None

    def create_webdriver(self):
        """
        Helper function that creates and returns a Selenium webdriver for the
        selected browser.
        """
        if self.browser == "Chrome":
            return webdriver.Chrome()
        elif self.browser == "Firefox":
            return webdriver.Firefox()
        elif self.browser == "Safari":
            return webdriver.Safari()
        elif self.browser == "Edge":
            return webdriver.Edge()
        else:
            return None

    def get_url(self, url):
        """
        Return contents of the URL.  Uses Selenium if browser is not Headless.
        """
        if self.browser is None or self.browser == "Headless":
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers)

            return response.text

        else:
            if self.webdriver is None:
                self.webdriver = self.create_webdriver()
            else:
                # Check if the web browser is still alive.  If not, start a new one.
                try:
                    title = self.webdriver.title
                except:
                    self.webdriver = self.create_webdriver()

            self.webdriver.get(url)
            return self.webdriver.page_source

    def calculate(self, expression):
        """
        Tool for evaluating mathmatical expressions
        """
        expression = expression.replace("^", "**")  # The LLM sometimes uses ^ for exponentation. Replace this operator with **

        try:
            return str(safe_eval(expression))
        except:
            return "That was not a valid expression"

    def google_search(self, topic):
        """
        Tool for using Google to search the web.
        Returns distilled HTML of Google search results with links included.
        """
        full_html = self.get_url("https://www.google.com/search?q=" + urllib.parse.quote(topic))

        html = distill_html(full_html)

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

    def get_web_page(self, url):
        """
        Tool for getting the contents of a web page.
        Returns distilled HTML with all links removed.
        """
        try:
            full_html = self.get_url(url)
            return distill_html(full_html, remove_links=True)
        except:
            return "Error retrieving web page"

