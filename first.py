import streamlit as st
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

# title
st.set_page_config(page_title="PeriodCare AI", layout="wide")

# Initialization
if "page" not in st.session_state:
    st.session_state.page = "Home"

# change pages
def change_page(page):
    st.session_state.page = page

st.markdown(
    """
    <style>
        .nav-button {
            display: inline-block;
            background-color:#786f80;
            color: white;
            padding: 12px 20px;
            margin: 5px;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
            border: none;
        }
        .stButton>button{
            background-color:#008ECC;
            color: #000000;
            padding: 12px 20px;
        }
        .stButton>button:hover{
            background-color:#000000;
            padding: 12px 20px;
        }
        .nav-button:hover {
            background-color: #e63766;
        }
        .sub-text {
            color: #555;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


#home page

if st.session_state.page == "Home":
    st.markdown('<h1 style="text-align:center; color:#FF4F66; font-size:70px;">PeriodCare AI</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center; color:#D03D56; font-size:40px;">Empowering Women with Period Care Tips!</h3>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-style:italic; color:#A280CC; font-size:20px;">"Personalized care for your unique menstrual health needs."</p>', unsafe_allow_html=True)

    st.markdown("""
        <style>
            .hero-text {
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                color: #D63384;
            }
            .sub-text {
                text-align: center;
                font-size: 20px;
                color: #555;
            }
            .btn-container {
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }
            .box {
                padding: 15px;
                border-radius: 10px;
                background-color: #000000;
                border: 1px solid #ddd;
                margin-bottom: 20px;
                font-size: 15px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("---")  

    # Features Section
    st.subheader("Why to Choose PeriodCare AI?")
    st.markdown('<div class="box"><h6>Managing menstrual health effectively requires the right guidance, which PeriodCare AI provides through AI-driven recommendations.<br>'
                '<h3 style="font-size:1rem;">♀️ PeriodCare AI is here to help...</h3><ul><li> Personalized product recommendations for your unique needs.</li><li>Exercise tips to reduce period pain and improve circulation.</li><li> Diet advice to support hormonal balance and overall well-being.</li><br>'
                '🎗️ Tailored support for everyone:<ul><li>Addresses concerns like PCOS, PMDD, and endometriosis.</li><li>Ensures comfort, safety, and the right choices for your body.</li><br>'
                '✔ Breaking taboos & spreading awareness <ul><li>Encourages open discussions about menstrual health.</li><li>Empowers individuals to make informed choices.</li>'
                ' <br>Because your menstrual health matters! 💜</p></h6></div>', unsafe_allow_html=True)

    col1, col2= st.columns(2)
    with col1:
        st.write("⭐ Get the best menstrual care products tailored to your needs.")
        st.write("Menstrual health🩸 is often overlooked, but it plays a crucial role in overall well-being. PeriodCare AI is an advanced, AI-powered solution designed to provide personalized menstrual health product recommendations tailored to your body’s unique needs. By analyzing key health indicators, it suggests the most suitable products, ensuring comfort, safety, and protection against potential infections.")
        if st.button("Start"):
            st.info("Going to product recommendation page...")
            change_page("Product Recommendation")
    with col2:
        st.write("⭐ Exercise and nutrition guidance to reduce discomfort and enhance well-being.")
        st.markdown('Beyond product recommendations, PeriodCare AI addresses common menstrual health concerns, offering targeted suggestions for conditions like PCOS, PMS, PMDD, and endometriosis. It also provides curated exercise tips to help reduce period pain, improve circulation, and promote relaxation. Additionally, personalized dietary guidance supports hormonal balance and enhances overall health.')
        if st.button("Visit"):
            st.info("Get personalised product recommendation page...")
            change_page("Personalized Product")

    st.markdown("---") 

    with st.expander("⭐ Awareness & Breaking the Taboo"):
        st.markdown("### ⭐ Awareness & Breaking the Taboo")
        st.write("PeriodCare AI is committed to breaking societal taboos surrounding menstruation by fostering awareness and education. We are redefining menstrual care with smart, science-backed insights designed for every body.")
        if st.button("Navigate"):
            st.info("Learn about Menstrual health")
            change_page("Awareness")


#product page
elif st.session_state.page == "Product Recommendation":
    st.markdown("""
        <style>
            body {
                background-color: #080808;
                text-size-adjust: 120%;
            }
            .title {
                color: #FC6C87;
                text-align: center;
                font-size: 40px;
                font-weight: bold;
            }
            .stButton>button {
                background-color:#B76E79;
                color: black;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }
            .stRadioButton label {
                font-size: 25px;
            }
            .stButton>button:hover {
                background-color: #ffffff;
            }
                
            div[class*="stRadio"] label {
                color: #ff477e !important;
                font-weight: bold;
            }
            div[class*="stRadio"] div[role="radiogroup"] {
                gap: 10px;
            }
            
            div[class*="stRadio"] label[data-baseweb="radio"]:hover {
                background-color: #786F80;
                padding: 5px 10px;
                border-radius: 10px;
            }
                
        </style>
    """, unsafe_allow_html=True)

    df = pd.read_csv("menstrual_product_recommendation.csv")
    categorical_columns = ["Age_group", "Pain_Level", "Eco-Friendly", "Skin_Sensitivity"]

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  

    df["Cost"] = df["Cost"].astype(float)
    product_encoder = LabelEncoder()
    df["Product_encoded"] = product_encoder.fit_transform(df["Product"])
    df_features = df[["Cost"] + categorical_columns]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)
    X_train, X_test, y_train, y_test = train_test_split(df_scaled, df["Product_encoded"], test_size=0.2, random_state=42)

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(X_train)

    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    # SVM Accuracy
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Hybrid 
    def hybrid_recommendation(user_input, n_neighbors=3):
        user_array = np.array(user_input).reshape(1, -1)
        user_scaled = scaler.transform(user_array)

        # KNN
        distances, indices = knn_model.kneighbors(user_scaled, n_neighbors=n_neighbors + 1)
        knn_recommendations = df.iloc[indices.flatten()]["Product_encoded"][1:].tolist()

        # SVM for best match
        svm_scores = svm_model.predict_proba(user_scaled)[0]
        ranked_products = sorted(zip(knn_recommendations, svm_scores), key=lambda x: -x[1])

        unique_products = list(dict.fromkeys([product_encoder.inverse_transform([p])[0] for p, _ in ranked_products]))[:2]

        return unique_products

    #  Streamlit UI
    st.markdown('<p class="title" style="font-size:58px;">PeriodCare AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="title">Menstrual Product Recommendation System</p>', unsafe_allow_html=True)
    st.write("Get personalized menstrual product recommendations based on your preferences!")


    age_labels = {0: "Teen", 1: "Young Adult", 2: "Adult", 3: "Middle-aged", 4: "Senior"}
    pain_labels = {0: "No", 1: "Yes"}
    eco_labels = {0: "Not Eco-Friendly", 1: "Eco-Friendly"}

    cost = st.number_input(
        "Cost",
        min_value=float(df["Cost"].min()), 
        max_value=float(df["Cost"].max()), 
        value=float(df["Cost"].median()),  
        step=10.0  
    )

    # Input
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.radio("Age Group", [0, 1, 2, 3, 4], format_func=lambda x: ["Teen", "Young Adult", "Adult", "Middle-aged", "Senior"][x])
    with col2:
        pain = st.radio("Pain Level", [0, 1], format_func=lambda x: "No Pain" if x == 0 else "Yes")
    with col3:
        sensitivity = st.radio("Skin Sensitivity", [0, 1, 2, 3], format_func=lambda x: f"Level {x}")
    with col4:
        eco_friendly = st.radio("Eco-Friendly", [0, 1], format_func=lambda x: "Not Eco-Friendly" if x == 0 else "Eco-Friendly")

    user_input = [cost, age, pain, eco_friendly, sensitivity]

    if st.button("Get Recommendations"):
        st.image("pic5.jpg",width = 100)
        recommendations = hybrid_recommendation(user_input)
        st.subheader("Recommended Products:")
        for i, product in enumerate(recommendations, 1):
            st.write(f"{i}. {product}")
    st.markdown("---")


# menstrual concern page
elif st.session_state.page == "Product Recommendation based on your Menstrual Health Concerns":

    # defining menstrual products for health related issues
    health_issues = {
        "PCOS": ["Organic Cotton Pads", "Menstrual Cups"],
        "Vaginal Infections": ["Unscented Cotton Pads", "pH-Balanced Period Wash"],
        "PMS": [],
        "PMDD": [],
        "Menorrhagia (Heavy Flow)": ["Super-Absorbent Pads", "High-Capacity Menstrual Cups", "Period Panties"],
        "Dysmenorrhea (Painful Cramps)": ["Menstrual Cup with Soft Silicone", "Ultra Soft Pads"]
    }

    # tips to do when suffering from these diseases
    tips = {
        "PCOS": [
            "Eat a low-carb, high-fiber diet like whole grains, green leafy vegetables, nuts.",
            "Include healthy fats like avocados, olive oil, flaxseeds for hormone balance.",
            "Avoid processed foods, sugar, and dairy to reduce insulin resistance.",
            "Inositol & Omega-3 can help with insulin resistance."
        ],
        "Vaginal Infections": [
            "Wear cotton underwear and avoid tight clothing.",
            "Avoid excess sugar, as it promotes bacterial growth. Eat protein-rich food.",
            "Avoid douching and use pH-balanced intimate washes."
        ],
        "Dysmenorrhea (Painful Cramps)": [
            "Apply a heating pad to the lower abdomen.",
            "Drink ginger tea or chamomile tea.",
            "Increase Omega-3-rich foods like salmon, walnuts, flaxseeds."
        ],
        "PMS": [
            "Practice mindfulness, deep breathing, and meditation.",
            "Increase calcium intake like yogurt, milk, leafy vegetables to ease mood swings.",
            "Cut down on sugar, caffeine, and processed foods."
        ],
        "PMDD": [
            "Get enough sleep (7-9 hours) to regulate hormones.",
            "Reduce stress levels through breathing exercises & mindfulness.",
            "Take vitamin B6 & calcium for mood regulation."
        ],
        "Menorrhagia (Heavy Flow)": [
            "Stay hydrated drink plenty of water.",
            "Try heat therapy for cramps and discomfort.",
            "Include iron-rich foods to prevent anemia.",
            "Avoid caffeine & salty foods to reduce bloating.",
            "Consider yoga or light exercises to improve circulation.",
            "Consult a doctor if bleeding lasts more than 7 days."
        ]
    }

    # Best Foods for Menstrual Health
    nutrition_recommendations = {
        "Iron-Rich Foods": ["Spinach", "Lentils", "Pumpkin Seeds", "Red Meat", "Tofu"],
        "Magnesium Sources": ["Bananas", "Almonds", "Spinach", "Avocados"],
        "Anti-Inflammatory Foods": ["Turmeric", "Ginger Tea", "Salmon", "Berries"],
        "Hydration Boosters": ["Coconut Water", "Herbal Teas", "Cucumber", "Watermelon"],
        "Hormonal Balance Foods": ["Flaxseeds", "Chia Seeds", "Yogurt", "Broccoli"]
    }

    # designing UI

    st.markdown("""
        <style>
            body {
                background-color: #080808;
                text-size-adjust: 120%;
            }
            .title {
                color: #63C5DA;
                text-align: center;
                font-size: 50px;
                font-weight: bold;
            }
            .subtitle {
                color: #FFFFFF;
                text-align: center;
                font-size: 20px;
            }
            .issue {
                background: #CC5500;
                padding: 20px;
                color : #ffffff;
                border-radius: 10px;
                box-shadow: 0px 4px 8px #C0C0C0;
                margin-bottom: 20px;
            }
            .box {
                background: #00316E;
                color: #FFFFFF;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                display: inline-block;
                box-shadow: 0px 2px 4px #9ECAE1;
            }
            .highlight {
                display: inline-block;
                background: #025043;
                color: white;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 16px;
                margin: 5px;
                font-weight: bold;
                box-shadow: 0px 4px 6px #234F1E;
            }
            .product-badge {
                display: inline-block;
                background: #CC5500;
                color: white;
                padding: 10px 15px;
                border-radius: 50px;
                font-size: 16px;
                margin: 5px;
                font-weight: bold;
                box-shadow: 0px 4px 6px #ED9121;
            }
            .stButton>button {
                background-color: #cc7722;
                color: black;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }
            .stButton>button:hover {
                background-color: #ffffff;
            }
        </style>
    """, unsafe_allow_html=True)


    st.markdown('<p class="title">PeriodCare AI for Personalised Health Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Get personalised recommendations based on your Menstrual Health Concerns</p>', unsafe_allow_html=True)

    st.markdown('<div class="box"><h4 style="color:white;">Select Your Menstrual Health Issues:</h4></div>', unsafe_allow_html=True)
    selected_issues = st.multiselect("", list(health_issues.keys()), max_selections=3)

    # Recommendations
    if selected_issues:
        for issue in selected_issues:
            st.markdown(f'<div class="issue"><h4  style="color:white;"> {issue}</h4>', unsafe_allow_html=True)

            if health_issues[issue]:
                st.markdown('<div class="box"><h4  style="color:white;"> Recommended Products:</h4>', unsafe_allow_html=True)
                product_html = "".join(f'<span class="product-badge">{product}</span>' for product in health_issues[issue])
                st.markdown(product_html, unsafe_allow_html=True)
            if issue in tips:
                st.markdown('<div class="box"><h4  style="color:white;">📌 Tips to Manage:</h4>', unsafe_allow_html=True)
                for tip in tips[issue]:
                    st.markdown(f'<div class="highlight"> {tip}</div>', unsafe_allow_html=True)


            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="box"><h4 style="color:white;">🥗 Best Foods for Menstrual Health</h4></div>', unsafe_allow_html=True)
        for category, foods in nutrition_recommendations.items():
            st.markdown(f'<div class="highlight"><b>{category}:</b> {", ".join(foods)}</div>', unsafe_allow_html=True)

        st.info("🩺 Please consult a doctor for personalized advice. ")
    else:
        st.warning("Please select at least one Menstrual Health if you don't, go to Product Recommendation page.")
    st.markdown("---")


# exercise page
elif st.session_state.page == "Exercise Recommendation":
    df = pd.read_csv("excercise.csv")
    original_df = df.copy()

    label_encoders = {}
    categorical_cols = ["Age Group", "Flow Level", "Health Condition", "Pain Level"]
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[categorical_cols])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    target_k = 18
    kmeans = KMeans(n_clusters=target_k, random_state=42, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

    hc = AgglomerativeClustering(n_clusters=target_k, linkage='ward')
    df["HC_Cluster"] = hc.fit_predict(X_scaled)

    #silhouette_kmeans = silhouette_score(X_pca, df["KMeans_Cluster"])
    #silhouette_hc = silhouette_score(X_pca, df["HC_Cluster"])

    exercise_mapping = {
        0: "Yoga", 1: "Stretching", 2: "Walking", 3: "Breathing Exercises",
        4: "Light Yoga", 5: "Moderate Cardio", 6: "Strength Training",
        7: "Pilates", 8: "Low-impact Workout", 9: "Tai Chi",
        10: "Cycling", 11: "Dancing", 12: "Swimming",
        13: "Meditation", 14: "Core Exercises", 15: "High-intensity Workout",
        16: "Foam Rolling", 17: "Balance Training"
    }

    def predict_cluster(age, flow, health, pain):
        input_data = np.array([[label_encoders["Age Group"].transform([age])[0],
                                label_encoders["Flow Level"].transform([flow])[0],
                                label_encoders["Health Condition"].transform([health])[0],
                                label_encoders["Pain Level"].transform([pain])[0]]])

        input_scaled = scaler.transform(input_data)
        #kmeans
        cluster_kmeans = kmeans.predict(input_scaled)[0]

        # HC act closest K-Means cluster
        closest_hc_cluster = df[df["KMeans_Cluster"] == cluster_kmeans]["HC_Cluster"].mode()[0]

        return closest_hc_cluster, exercise_mapping.get(closest_hc_cluster, "General Exercise")


    # UI
    st.markdown("""
        <style>
            body { 
                background-color: #050505; 
                color: white; 
            }
            .stButton>button {
                background-color: #663d65;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }
            .stButton>button:hover { 
                background-color: #ffffff; 
            }
            .stSelectbox label { 
                font-size: 22px; 
                font-weight: bold; 
            }
            .stTitle {
                font-size: 48px !important;
                font-weight: bold;
                text-align: center;
                color:#6F2DA8;
            }
            .stSubtitle {
                font-size: 24px;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    # UI Header
    st.markdown('<p class="stTitle">PeriodCare AI Menstrual Exercise Recommendation</p>', unsafe_allow_html=True)
    st.markdown('<p class="stSubtitle">Enter your details for personalized exercise recommendation...</p>', unsafe_allow_html=True)

    # UI Inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox("Select Age Group", ["18-24", "25-34", "35-45"])
        flow = st.selectbox("Select Flow Level", ["Light", "Moderate", "Heavy"])

    with col2:
        health = st.selectbox("Select Health Condition", ["PCOS", "Endometriosis", "PMDD", "None"])
        pain = st.selectbox("Select Pain Level", ["Mild", "Moderate", "Severe"])

    # UI Button
    if st.button("Get Recommendation"):
        cluster, exercise = predict_cluster(age, flow, health, pain)
        st.success(f"{exercise} is best in these days.")

    # UI Image Display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("pic1.png", width=300)
    with col2:
        st.image("pic2.png", width=300)
    with col3:
        st.image("pic3.png", width=300)
    with col4:
        st.image("pic4.png", width=200)
    st.markdown("---")


# awareness
elif st.session_state.page == "Awareness":
    file_path = "awarness.csv"  
    df = pd.read_csv(file_path)

    st.markdown (
        """
        <style>
        .stTitle {
                font-size: 48px !important;
                font-weight: bold;
                text-align: center;
                color: #97456C;
        }
        .stButton>button {
            background-color: #043927;
            color:#ffffff;
        }
        .stButton>button:hover {
            background-color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True
    )

    questions = df["instruction (string)"].tolist()
    answers = df["output (string)"].tolist()

    st.markdown('<p class="stTitle">#PeriodTips for Menstrual Health Awareness</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
            st.image("pic7.png",width=400);
    with col2:
            st.image("pic9.png",width=356);
    with col3:
            st.image("pic8.png",width=414);
    
    selected_questions = st.multiselect(
        "What's your question?: ", questions
    )

    if selected_questions:
        st.write("### Answers:")
        for question in selected_questions:
            answer = df[df["instruction (string)"] == question]["output (string)"].values[0]
            st.write(f"Question : **{question}**")
            st.write(f"Answer : {answer}\n")
        
        
        
    st.markdown("---")

# Navigation Bar
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("Home"):
        change_page("Home")
with col2:
    if st.button("Product Recommendation"):
        change_page("Product Recommendation")
with col3:
    if st.button("Personalized Product"):
        change_page("Product Recommendation based on your Menstrual Health Concerns")
with col4:
    if st.button("Exercise"):
        change_page("Exercise Recommendation")
with col5:
    if st.button("Awareness"):
        change_page("Awareness")
st.markdown('</div>', unsafe_allow_html=True)




st.markdown("___")
st.markdown("<p class='sub-text'><b>Get your Personalised Recommendation here...</b></p>", unsafe_allow_html=True)
st.markdown("No copyright @ PeriodCare AI")


