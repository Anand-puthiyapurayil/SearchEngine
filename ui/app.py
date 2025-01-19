# project/ui/app.py
import streamlit as st
import requests

st.title("Semantic Search UI")

query = st.text_input("Enter your query (e.g., 'cheap pumps', 'pump services'):")
top_k = st.number_input("Number of results:", min_value=1, max_value=50, value=10)

if st.button("Search"):
    resp = requests.post("http://localhost:8000/api/search", json={"query": query, "top_k": top_k})
    if resp.status_code == 200:
        data = resp.json()["results"]
        for r in data:
            st.write("------")
            r_type = r.get("type", "unknown")

            if r_type == "product":
                # Show product fields: product_name, price, rating
                st.subheader(r.get("product_name", "Unknown Product"))
                if "price" in r:
                    st.write(f"Price: {r['price']}")
                if "rating" in r:
                    st.write(f"Rating: {r['rating']}")
            else:
                # Service fields: service_name, price_range
                st.subheader(r.get("service_name", "Unknown Service"))
                if "price_range" in r:
                    st.write(f"Price Range: {r['price_range']}")

            st.write(f"Description: {r.get('description', 'No description')}")
            st.write(f"Similarity: {r['similarity']}")
            st.write(f"Type: {r_type}")
    else:
        st.error("No results found or error occurred.")
