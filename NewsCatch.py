import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 爬取BBC Earth领域的新闻
def crawl_bbc_earth():
    url = "https://www.bbc.co.uk/news/science_and_environment"
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, 'html.parser')

    news_articles = soup.find_all('h3', class_='gs-c-promo-heading__title gel-pica-bold nw-o-link-split__text')

    # 提取新闻标题作为关键字
    keywords = [news.text for news in news_articles]
    return keywords

# 计算TF-IDF
def calculate_tfidf(keywords):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(keywords)
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df

# 主函数
def main():
    keywords = crawl_bbc_earth()
    tfidf_table = calculate_tfidf(keywords)
    print(tfidf_table)

if __name__ == "__main__":
    main()


