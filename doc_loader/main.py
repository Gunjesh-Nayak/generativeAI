from langchain_community.document_loaders import WebBaseLoader

url=["https://www.who.int/data/GIS","https://www.who.int/data/data-collection-tools/harmonized-health-facility-assessment/introduction"]
loader = WebBaseLoader(url)
documents = loader.load()
print(documents[1].page_content)