import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_words_uusikielemme(
        list_url: list[str],
)->pd.DataFrame:
    # get tables
    finn = []
    eng = []
    # get parent links
    for url in list_url:
        print(f"Fetching {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
    
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')  # 
                row_data = [col.get_text(strip=True) for col in cols]
                if len(row_data)>=2:
                    finn.append(row_data[0])
                    eng.append(row_data[1])
        # dictionary 
        fe_dict = {
            "Word" : finn,
            "Definition" : eng
        }
        df_fe = pd.DataFrame(fe_dict)
    return df_fe                 

# get dataframe from commonlyusedwords.com
list_url = ["https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/fruit-hedelmat-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/vegetables-vihannekset-tai-kasvikset",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/flowers-kukat",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/fish-and-sea-creatures-kalat-ja-merenelavat",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/animal-body-parts-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/mammals-animals-elaimet-nisakkaat",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/insects-animals-elaimet-hyonteiset",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/berries-marjat-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/herbs-and-spices-yrtit-ja-mausteet",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/dishes-and-kitchen-appliances-keittiossa",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/meat-liha-ja-liharuokia-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/clothes-vaatteet-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/at-the-store-kaupassa",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/at-the-bank-pankissa-finnish-bank-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/talking-about-the-weather-ilma-saa-keli",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/finnish-winter-vocabulary-talvi",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/country-names-in-finnish-mista-sina-olet-kotoisin",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/school-koulu-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/mathematics-matematiikka",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/physics-fysiikka",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/birds-lintu-linnut-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/bodyparts-kehonosat",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/breakfast-aamiainen-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/list-of-hobbies-in-finnish-harrastukset",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/sauna-kiuas-lauteet-ja-hoyry-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/gardening-puutarhanhoito",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/vacationing-matkustaminen-loma-lomalla",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/professions-ammatit-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/describing-character-personality-luonne-persoonallisuus",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/friendship-and-love-ystavyys-ja-rakkaus",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/asuminen-kerrostalossa-living-in-finland-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/tools-tyokalut-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/cleaning-siivoaminen-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/traffic-liikenne",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/means-of-transportation",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-phrases/asking-for-directions-in-finnish-tien-neuvominen",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/music-musiikki-millaisesta-musiikista-pidat",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/puhelin-kannykka-finnish-telephone-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/computer-tietokone-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/at-the-library-kirjastossa",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/at-the-store-kaupassa",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/auton-osat-car-parts-finnish-vocabulary",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/finnish-postal-services-vocabulary-posti-kirje-paketti",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/remontti-finnish-vocabulary-related-to-renovating",
      "https://uusikielemme.fi/finnish-vocabulary/vocabulary-lists/childhood-lapsuus-finnish-vocabulary"
      ]

df = get_words_uusikielemme(list_url)
df.to_csv('data/dictionary.csv', index=False)



