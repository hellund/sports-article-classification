from bs4 import BeautifulSoup as soup
import requests
from datetime import date, timedelta
import pandas as pd
from tqdm import tqdm
import re
from src.slack_alert.sofus_alert import sofus_alert


def create_date_interval_df(start_year: int, start_month: int,
                            start_day: int, end_year: int,
                            end_month: int, end_day: int) -> pd.DataFrame:
    """Create a DataFrame with dates from start of interval to end.

    Args:
        start_year (int): start year
        start_month (int): start month
        start_day (int): start day
        end_year (int): end year
        end_month (int): end month
        end_day (int): end day

    Returns:
        pd.DataFrame: DataFrame containing dates from start of interval to end
    """
    first_date = date(start_year, start_month, start_day)
    last_date = date(end_year, end_month, end_day)
    dates = pd.date_range(first_date, last_date - timedelta(days=1), freq='d')
    dates = [str(day) for day in dates]
    date_frame = pd.DataFrame(dates, columns=['date'])
    date_frame[['date', 'time']] = date_frame['date'].str.split(' ',
                                                                expand=True)
    date_frame[['year', 'month', 'day']] = date_frame['date'].str.\
        split('-', expand=True)
    return date_frame


def find_relevant_articles(date_frame: pd.DataFrame) -> pd.DataFrame:
    """Scrapes VG.no for relevant articles of football news and creates a
    DataFrame containing the title, url and date of those relevant articles

    Args:
        date_frame (pd.DataFrame): DataFrame containing dates for collecting
        articles.

    Returns:
        pd.DataFrame: DataFrame containing title, url and date of relevant
        articles
    """
    relevant_articles = []
    for _, row in tqdm(date_frame.iterrows(), total=date_frame.shape[0]):
        url = "https://www.vg.no/sport/fotball?before={}-{}-{}T{}Z". \
            format(row['year'], row['month'], row['day'], row['time'])
        html = requests.get(url)
        bs_object = soup(html.content, 'lxml')
        for tag in bs_object.findAll('a', {'class': 'nolinkstyles hyperion-css-1s8spa1'}):
            relevant_articles.append(
                [tag['aria-label'], 'https://vg.no' + tag['href'], row['date']])
    df_relevant_articles = pd.DataFrame(relevant_articles, columns=[
        'heading', 'href', 'date'])
    df_relevant_articles.to_csv('relevant_urls.csv', encoding="utf-8-sig")
    df_relevant_articles = df_relevant_articles.drop_duplicates(
        subset=['heading'],
        keep='first')
    df_relevant_articles.reset_index(drop=True, inplace=True)
    return df_relevant_articles


def remove_html_tags(text: str) -> str:
    """Remove html tags from a text string.

    Args:
        text (str): text as string containing html tags

    Returns:
        str: cleaned text string without html tags
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def find_tag(text: str) -> str:
    """Finds the tags from the start of a html-tag string.

    Args:
        text (str): text as string containing html tags

    Returns:
        str: string of two first characters ment to represent type of html tag
    """
    text = text[1:3]
    text = text.strip()
    return text


def scrape_articles_from_url_df(url_df: pd.DataFrame) -> pd.DataFrame:
    """Scraping articles from VG.no using the urls from the supplied
    DataFrame and perform som initial processing to create a complete df.

    Args:
        url_df (pd.DataFrame): DataFrame containing the urls of the articles
        to scrape.

    Returns:
        pd.DataFrame: DataFrame containing html tags scraped from the urls.
    """
    article_texts = []
    for idx, row in tqdm(url_df.iterrows(), total=url_df.shape[0]):
        page = requests.get(row['href'])
        bs_object = soup(page.content, 'lxml')
        try:
            for news in bs_object.find('article').findAll(
                    ['h1', 'h2', 'h3', 'p']):
                article_texts.append([news,
                                      row['heading'],
                                      row['href'],
                                      row['date'],
                                      idx])
        except:
            article_texts.append('')

    article_df = pd.DataFrame(article_texts, columns=['text', 'heading',
                                                      'href', 'date',
                                                      'article_index'])
    article_df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    article_df['tag'] = article_df['text'].apply(
        lambda text: find_tag(str(text)))
    article_df['text'] = article_df['text'].apply(
        lambda text: remove_html_tags(str(text)))
    return article_df


def convert_df_to_csv(article_df: pd.DataFrame, path: str) -> None:
    """

    Args:
        article_df (pd.DataFrame): DataFrame that is to be exported as csv
        path (str): string containing the path where the csv should be saved

    Returns:
        NoneType
    """
    article_df.to_csv(path, encoding="utf-8-sig", index=False)


def main() -> None:
    """Function that runs when __name__ == '__main__' that scrapes football
    articles from VG.no posted between 01.01.2022 and 01.01.2023 and creates
    a csv file of the paragraphs found in those articles.

    Returns:
        NoneType
    """
    date_frame = create_date_interval_df(2022, 1, 1, 2023, 1, 1)
    url_df = find_relevant_articles(date_frame)
    article_df = scrape_articles_from_url_df(url_df)
    convert_df_to_csv(article_df, 'vg_articles_2022.csv')
    # article_df = pd.read_csv('vg_articles_2022.csv', encoding='utf-8-sig')
    sofus_alert()


if __name__ == '__main__':
    main()
