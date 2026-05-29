# **Finnish - English Vocabulary Trainer App**

## **Overview**

- **App name:** Suomi Sanasto  
- **Purpose:** Learning Finnish vocabularies  
- **Author:** Anh Dung Le  
- **License:** ©️ CC0: Public Domain 
- **Date:** May 24, 2026


## **Project Directory Structure**

<pre>
<b>finnish_learn</b>/
├──  🐍<span style="color:orange"> app.py </span>
├──  🐍<span style="color:orange"> get_dict.py </span>  
├──  📝<span style="color:orange"> readme.md </span>  
├──  📋<span style="color:orange"> requirements.txt </span>    
├──  🗂️ data/  
     ├── 𝄜 <span style="color:orange">dictionary.csv </span>  
     ├── {} <span style="color:orange">stats.json </span>  
├──  🗂️ templates/  
     ├── <span style="color:orange">layout.html</span>  
     ├── <span style="color:orange">index.html</span>  
     ├── <span style="color:orange">learn.html</span>  
     ├── <span style="color:orange">flashcard.html</span>  
     ├── <span style="color:orange">mc.html</span>  
     ├── <span style="color:orange">typing.html</span>  
     └── <span style="color:orange">stats.html</span>
</pre>


## **Content**
1. <b>`get_dict.py`</b>: scrapes data (Finnish words) from the website [uusikielemme.fi](https://uusikielemme.fi/). Scrapped data is then stored in <b>`data/dictionary.csv`</b>.

2. <b>`app.py`</b>: runs the server, reads the CSV, and manages <b>`data/stats.json`</b>.

3.  The HTML templates stored in the folder <b>`templates/`</b>

4. <b>`requirements.txt`</b>: Python requirements for running the app

5. <b>`readme.md`</b>: README file (this file)

## **Preparation**

1. Install Python3
2. Install requirements 

    ```
    pip install -r requirements.txt
    ```
## **How to run the application?**

1. Getting the dictionary
    ~~~
    python3 get_dict.py
    ~~~
2. Running the app 
    ~~~
    python3 app.py
    ~~~
3. Open your web browser and go to: http://127.0.0.1:5000

## **App content**

1. Learning mode
2. Flashcard mode
3. Quiz: 
    - Multiple-Choice Questions
    - Typing Questions 
4. Quiz Statistics 

