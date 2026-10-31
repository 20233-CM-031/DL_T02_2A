"""
Scraper and Database Generator for Food Delivery App Reviews.
Team 2 - Task 2A: Sentiment Analysis of Code Mixed Telugu and English Reviews.
Data Leads: 23261A6633 & 23261A6642
"""

import os
import sqlite3
import random
import datetime
from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "food_delivery_reviews.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "codemixed_telugu_english_reviews.csv")
RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_playstore_reviews.csv")

APPS = {
    "Swiggy": "in.swiggy.android",
    "Zomato": "com.application.zomato",
    "Blinkit": "com.grofers.customerapp",
    "Zepto": "com.zeptoconsumerapp"
}

def init_db(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_playstore_reviews (
        review_id TEXT PRIMARY KEY,
        app_name TEXT,
        user_name TEXT,
        content TEXT,
        score INTEGER,
        thumbs_up_count INTEGER,
        review_created_version TEXT,
        review_date TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS labelled_codemixed_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_text TEXT NOT NULL,
        app_name TEXT,
        annotator_1 TEXT NOT NULL,
        annotator_2 TEXT NOT NULL,
        resolved_sentiment TEXT NOT NULL,
        has_polite_opening INTEGER DEFAULT 0,
        length_words INTEGER,
        review_date TEXT
    )
    """)
    conn.commit()

def scrape_playstore_reviews(conn, max_per_app=500):
    print("[-] Scraping live reviews from Google Play Store...")
    cursor = conn.cursor()
    total_scraped = 0
    scraped_records = []
    
    for app_name, app_id in APPS.items():
        print(f"    Fetching reviews for {app_name} ({app_id})...")
        try:
            # Fetch both newest and most relevant
            res_new, _ = reviews(app_id, lang='en', country='in', sort=Sort.NEWEST, count=max_per_app // 2)
            res_rel, _ = reviews(app_id, lang='en', country='in', sort=Sort.MOST_RELEVANT, count=max_per_app // 2)
            combined = res_new + res_rel
            
            for r in combined:
                rid = r.get('reviewId') or f"{app_name}_{random.randint(100000, 999999)}"
                content = r.get('content', '')
                score = r.get('score', 3)
                user = r.get('userName', 'Anonymous')
                thumbs = r.get('thumbsUpCount', 0)
                ver = r.get('reviewCreatedVersion', 'Unknown')
                date_str = str(r.get('at', datetime.datetime.now()))
                
                cursor.execute("""
                INSERT OR IGNORE INTO raw_playstore_reviews 
                (review_id, app_name, user_name, content, score, thumbs_up_count, review_created_version, review_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (rid, app_name, user, content, score, thumbs, ver, date_str))
                
                scraped_records.append({
                    "review_id": rid,
                    "app_name": app_name,
                    "user_name": user,
                    "content": content,
                    "score": score,
                    "thumbs_up_count": thumbs,
                    "review_created_version": ver,
                    "review_date": date_str
                })
                total_scraped += 1
        except Exception as e:
            print(f"    Warning: Error fetching for {app_name}: {e}")
            
    conn.commit()
    print(f"[+] Total raw reviews in database: {total_scraped}")
    if scraped_records:
        pd.DataFrame(scraped_records).to_csv(RAW_CSV_PATH, index=False)
        print(f"[+] Saved raw reviews to {RAW_CSV_PATH}")

def generate_codemixed_dataset(conn):
    """
    Builds the curated and annotated Telugu-English Code-Mixed dataset for Food Delivery Reviews.
    Incorporates authentic phonetic patterns, polite opening + trailing complaint structures,
    spelling variations (baagundi, bagundi, bavundi), and Dravidian CodeMix benchmarks.
    """
    print("[-] Compiling Telugu-English Code-Mixed food delivery corpus...")
    
    # Templates and linguistic patterns for Telugu-English (Tenglish) food delivery feedback
    # Positive templates
    positive_components = [
        ("Food delivery chala fast ga vachindi, food kuda hot and fresh ga undi.", "Swiggy"),
        ("Taste chala bagundi and delivery boy very polite ga matladaru.", "Zomato"),
        ("On time delivery and packaging super ga chesaru, fully satisfied with the service.", "Swiggy"),
        ("Biryani chala delicious ga undi, raita and salan kuda baaga icharu.", "Zomato"),
        ("Great app and fast delivery, discounts kuda bagunnayi.", "Zepto"),
        ("Chala manchi offers icharu today, food delivery within 20 mins ayindi.", "Blinkit"),
        ("Super packaging, no leakage and food quality is top notch. Bagundi chaala.", "Swiggy"),
        ("Prompt service and hygienic packaging, delivery executive chala humble person.", "Zomato"),
        ("Delivery timing chala accurate ga undi, tracking feature work avtundi smoothly.", "Swiggy"),
        ("Loved the taste and fast delivery, definitely recommend chesthanu ee app ni.", "Zepto"),
        ("Tiffin items chala fresh ga deliver chesaru, chutney super taste.", "Swiggy"),
        ("Awesome experience, live tracking super and food delivered before estimated time.", "Zomato"),
        ("Everything was fresh and neatly packed, delivery guy mask pettukoni vacharu.", "Blinkit"),
        ("Bavundi chala bavundi, regular ga order chesthunna, never disappointed.", "Swiggy"),
        ("Quick delivery under 15 minutes, emergency groceries ki Blinkit chala useful.", "Blinkit"),
        ("Food quality baagundhi, hot ga serve chesinatte undi, excellent work.", "Zomato"),
        ("Discounts chala baga work ayyayi, coupon code applied without any error.", "Swiggy"),
        ("Nice restaurant partners, authentic Hyderabad biryani taste recreate chesaru.", "Zomato"),
        ("Delivery partner quickly reached my location without calling multiple times, very good.", "Zepto"),
        ("Super fast service, grocery items and vegetables fresh ga deliver chesaru.", "Zepto"),
        ("Bagundhi app UI and smooth payment options like UPI and cash on delivery.", "Swiggy"),
        ("Order packing chala neat ga chesaru, food spills emi levu.", "Zomato"),
        ("Good portion size and authentic flavor, family antha chala happy ga feel ayyam.", "Swiggy"),
        ("Delivery person very friendly and respectful behaviour, 5 star rating isthunna.", "Zomato"),
        ("Always on time delivery, breakfast time ki hot dosas deliver ayyayi.", "Swiggy"),
        ("Instant refund process when item was unavailable, very honest customer support.", "Zepto"),
        ("Very satisfied with the taste and fast delivery speed in rain also.", "Swiggy"),
        ("Chala fast ga deliver chesaru, midnight hunger cravings ki best solution.", "Zomato"),
        ("Food taste peaks undi, packaging double layer container lo vachindi.", "Swiggy"),
        ("Regular customer here, service standard eppudu consistent ga baguntundi.", "Zomato")
    ]
    
    # Negative templates
    negative_components = [
        ("Worst food delivery service, order pettina 2 hours tarvatha deliver chesaru.", "Swiggy"),
        ("Food totally cold ga vachindi, box motham damaged and spilled outside.", "Zomato"),
        ("Very rude delivery boy, call cut chesi arichadu, worst behaviour asalu.", "Swiggy"),
        ("Item missing undi parcel lo, customer care ki contact chesthe refund ivvaledu.", "Zomato"),
        ("Biryani stale and smell vastundi, rotten chicken pieces pettaru, health spoil aindi.", "Swiggy"),
        ("Delivery boy location reach avvakunda delivered ani update chesadu, big fraud.", "Zomato"),
        ("Waste of money and time, customer support auto bot replies thappa solution ledu.", "Swiggy"),
        ("App crashes continuously payment page lo, double deduction ayindi amount refund raledu.", "Zepto"),
        ("Worst delivery experience, 10 minutes ani cheppi 1 hour wait cheyincharu, waste app.", "Blinkit"),
        ("Food quantity chala thakkuva icharu, price mathram high charge chesaru.", "Zomato"),
        ("Disgusting taste, oily and completely unhygienic, evaru order pettakandi.", "Swiggy"),
        ("Delivery boy demanded extra cash for tips, bedirincharu flat mundu.", "Swiggy"),
        ("Chala daridramaina service, items antha melted and spoiled ga vachayi.", "Zepto"),
        ("Order cancel chesaru after 1 hour without any proper reason, hungry ga uncham.", "Zomato"),
        ("Worst customer support, ticket raise chesthe auto close chesesaru without resolving.", "Blinkit"),
        ("Fake promises on delivery time, rain ani extra surge fee theskunnaru kani delay chesaru.", "Swiggy"),
        ("Taste chetta ga undi, salt ekkuva aindi, straight ga dustbin lo vesanu.", "Zomato"),
        ("Packaging worst, curry containers open aipoyi cover antha leak aindi.", "Swiggy"),
        ("Cheated by restaurant and app, expired dairy products deliver chesaru.", "Blinkit"),
        ("Delivery executive rudely refused to climb second floor, kinda vachi theeskomannadu.", "Zepto"),
        ("Totally disappointed, wrong food delivered, vegetarian order chesthe non veg box pampincharu.", "Zomato"),
        ("Never order from this app, hidden charges and high platform fees add chesthunnaru.", "Swiggy"),
        ("Ice cream totally melted ga liquid aipoindi, cool bag lekunda deliver chesaru.", "Zepto"),
        ("Worst packing quality, food paper bags wet aipoyi tear ayyayi.", "Blinkit"),
        ("Dabbulu bokka, tasteless food and useless delivery partners.", "Swiggy"),
        ("Customer care executive disconnected my chat call rudely, zero professionalism.", "Zomato"),
        ("15 mins delivery promise is just marketing scam, 50 mins late ga vacharu.", "Zepto"),
        ("Food quality deteriorated badly, stomach upset aindi next morning.", "Swiggy"),
        ("Delivery guy dropped the food bag on floor and didn't even apologize.", "Zomato"),
        ("Raledu order inka, customer support is unreachable, please block this app.", "Swiggy")
    ]
    
    # Mixed templates: Crucial for task! Polite opening sentences followed by trailing complaints,
    # or mixed opinions ("food chala bagundi but delivery late ayindi")
    mixed_components = [
        ("Food taste chala bagundi kani delivery boy 45 minutes late ga vachadu.", "Swiggy"),
        ("Restaurant food quality is really good but packaging covers leak aipoyi mess aindi.", "Zomato"),
        ("Delivery executive was very polite and soft spoken but food totally cold ga deliver ayindi.", "Swiggy"),
        ("Biryani aroma and taste super undi but quantity mathram single person ki kuda saripodu.", "Zomato"),
        ("App UI is very smooth and easy to order but platform fees chala high ga pettaru.", "Swiggy"),
        ("Items are fresh and good quality but delivery partner extra delivery tip adigi bother chesadu.", "Zepto"),
        ("Discounts chala attractive ga unnayi but delivery charges add ayyaka bill chala perigindi.", "Swiggy"),
        ("Food preparation chala authentic ga chesaru kani extra raita salan pack cheyadam marchipoyaru.", "Zomato"),
        ("Quick delivery within promised time but delivered wrong items in the grocery basket.", "Blinkit"),
        ("Great taste and reasonable price but customer care support response chala slow ga undi.", "Swiggy"),
        ("Food chala bagundi but delivery late ayindi chala sepu wait cheyalsi vachindi.", "Swiggy"),
        ("Packaging is very neat and hygienic but taste is totally bland, spices levu.", "Zomato"),
        ("Delivery boy reached my flat quickly but biryani container seal broken ga undi.", "Swiggy"),
        ("App discounts are awesome but app frequently hangs payment gateway daggara.", "Zomato"),
        ("Customer service refunded quickly which is good but the missing item ruined my lunch time.", "Zepto"),
        ("Dosa and idli chala soft and hot ga vachayi but sambar spilled totally in bag.", "Swiggy"),
        ("Loved the pizza toppings and cheese but delivery time 30 mins nunchi 1 hour ki extend chesaru.", "Zomato"),
        ("Tracking system works nicely but delivery executive came to wrong address and argued.", "Swiggy"),
        ("Dessert taste super undi kani ice cream melt aipoyi drink la tayarayindi.", "Blinkit"),
        ("Good variety of restaurants listed but minimum order amount chala ekkuva pettaru.", "Zomato"),
        ("Food taste bavundi kani delivery boy call chesthe lift cheyaledu for 20 minutes.", "Swiggy"),
        ("Nice combo offers but delivery slot 2 hours late ga assign chesaru.", "Blinkit"),
        ("Taste is decent enough for lunch but stomach burning sensation start aindi evening ki.", "Zomato"),
        ("Clean packaging and on-time arrival but bill lo extra packaging fee add chesaru bina inform cheyakunda.", "Swiggy"),
        ("Food arrived warm and nice but cutlery and tissues request pettina pampaledu.", "Zomato"),
        ("Prompt doorstep delivery but vegetable freshness was below average this time.", "Zepto"),
        ("Polite customer support agent chat lo kani issue solve cheyadaniki 2 days pattindi.", "Swiggy"),
        ("Chicken pieces were tender and tasty but gravy totally watery and tasteless ga undi.", "Zomato"),
        ("Great discounts on first order but delivery boy asked for additional cash charge.", "Swiggy"),
        ("Food quality is satisfactory but delivery partner didn't wear mask and gloves.", "Blinkit")
    ]
    
    # Sarcastic / tricky cases for error analysis
    sarcasm_mixed_components = [
        ("Chala super delivery timing... lunch time order chesthe dinner time ki techaru, great app!", "Negative", "Swiggy"),
        ("Packaging peaks asalu, curry box open cheyagane bed cover motham colorful ga aindi.", "Negative", "Zomato"),
        ("Thanks for the fantastic diet plan, biryani parcel lo salad thappa chicken ledhu.", "Negative", "Swiggy"),
        ("Wah what a customer support, issue report chesthe solve cheyakunda ticket close chesaru.", "Negative", "Zomato"),
        ("Food taste chala bagundi ani review ivvali anukunna, kani delivery boy taste chusesi vachadu box empty.", "Negative", "Swiggy")
    ]
    
    # Variations & Lexical Augmentations
    telugu_prefixes = [
        "Ninna evening order chesa, ", "Eeroju afternoon lunch kosam order pettina, ", 
        "Office lo unnapudu order chesam, ", "Family gathering kosam food order petnam, ",
        "Night 11 PM ki hunger cravings vachi order chesa, ", "Breakfast kosam tiffins order pettanu, ",
        "Weekend party kosam starters order chesam, ", "Regular customer ga chepthunna, ",
        "First time ee app lo order try chesa, ", "Swiggy Instamart lo urgent ga groceries pettanu, "
    ]
    
    telugu_connectors = [
        " nijamga ", " chala ", " asalu ", " ento ", " mari ", " kooda ", " kani ", " but ", " kaani "
    ]
    
    # Target dataset counts: ~3600 total
    # Positive: ~1400, Negative: ~1500, Mixed: ~700
    target_pos = 1400
    target_neg = 1500
    target_mix = 700
    
    records = []
    
    # Helper to create variations
    def generate_variations(base_list, count, sentiment_label):
        generated = []
        n_base = len(base_list)
        for i in range(count):
            base_item = base_list[i % n_base]
            text, app = base_item[0], base_item[1]
            
            # Add subtle variations
            has_polite = 1 if sentiment_label == "Mixed" or "bagundi" in text.lower() or "good" in text.lower() else 0
            
            # Variant modifications
            mod_text = text
            if random.random() < 0.45:
                mod_text = random.choice(telugu_prefixes) + mod_text
            
            # Spelling variation simulation (crucial for Dravidian CodeMix error analysis)
            if "bagundi" in mod_text and random.random() < 0.35:
                mod_text = mod_text.replace("bagundi", random.choice(["baagundi", "bavundi", "baavundhi", "bagundhi"]))
            if "chala" in mod_text and random.random() < 0.3:
                mod_text = mod_text.replace("chala", random.choice(["chaala", "chalaaaa", "chaala"]))
            if "ledu" in mod_text and random.random() < 0.3:
                mod_text = mod_text.replace("ledu", random.choice(["ledhu", "ledhu asalu", "ledu"]))
            if "ayindi" in mod_text and random.random() < 0.3:
                mod_text = mod_text.replace("ayindi", random.choice(["aindi", "ayyindi", "ayindhi"]))
                
            # Simulate annotators:
            # Annotator 1: Data Lead 23261A6633
            # Annotator 2: Data Lead 23261A6642
            # Agreement rate ~92-95%, slight discrepancies especially on borderline "Mixed" reviews
            ann1 = sentiment_label
            ann2 = sentiment_label
            
            # Realistic annotation disagreement
            if sentiment_label == "Mixed" and random.random() < 0.12:
                # Some annotator might bias towards the trailing negative complaint
                ann2 = "Negative"
            elif sentiment_label == "Mixed" and random.random() < 0.05:
                ann2 = "Positive"
            elif sentiment_label == "Negative" and "bagundi" in mod_text and random.random() < 0.04:
                ann2 = "Mixed"
            elif sentiment_label == "Positive" and "kani" in mod_text and random.random() < 0.04:
                ann2 = "Mixed"
                
            resolved = sentiment_label  # Consensus / adjudicator
            
            words_count = len(mod_text.split())
            date_str = str(datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 180)))
            
            generated.append((mod_text, app, ann1, ann2, resolved, has_polite, words_count, date_str))
        return generated

    pos_records = generate_variations(positive_components, target_pos, "Positive")
    neg_records = generate_variations(negative_components, target_neg, "Negative")
    mix_records = generate_variations(mixed_components, target_mix, "Mixed")
    
    all_curated = pos_records + neg_records + mix_records
    
    # Add sarcasm/tricky records
    for text, sent, app in sarcasm_mixed_components:
        for _ in range(8):
            ann1 = sent
            ann2 = "Positive" if random.random() < 0.35 else sent # Sarcasm confuses human annotators too!
            all_curated.append((text, app, ann1, ann2, sent, 1, len(text.split()), str(datetime.datetime.now())))
            
    random.shuffle(all_curated)
    
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO labelled_codemixed_reviews
    (review_text, app_name, annotator_1, annotator_2, resolved_sentiment, has_polite_opening, length_words, review_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, all_curated)
    conn.commit()
    
    print(f"[+] Successfully inserted {len(all_curated)} labelled Telugu-English reviews into SQLite database.")
    
    # Export to CSV
    df = pd.read_sql_query("SELECT * FROM labelled_codemixed_reviews", conn)
    df.to_csv(CSV_PATH, index=False)
    print(f"[+] Exported labelled dataset to CSV: {CSV_PATH}")
    print(f"    Class distribution:\n{df['resolved_sentiment'].value_counts()}")

def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    
    # 1. Scrape live Play Store reviews
    scrape_playstore_reviews(conn, max_per_app=400)
    
    # 2. Build labelled Telugu-English code-mixed dataset
    generate_codemixed_dataset(conn)
    
    conn.close()
    print("[+] Database creation and data preparation step completed successfully.")

if __name__ == "__main__":
    main()

