import streamlit as st
import pandas as pd
import sqlalchemy
from pandasai import SmartDataframe
from pandasai.connectors import PandasConnector
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
import os
import matplotlib

# If you wish to store your OpenAI key in an environment variable called OPENAI_API_KEY:
# os.environ["OPENAI_API_KEY"] = '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'


# Given MySQL Credentials
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "password"
MYSQL_DATABASE = "insight_ai"

class OutputParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def parse(self, result):
        if result['type'] == "dataframe":
            st.dataframe(result['value'])
        elif result['type'] == 'plot':
            st.image(result["value"])
        else:
            st.write(result['value'])
        return

def setup():
    custom_css = """
    <style>
    #MainMenu {visibility: hidden;}
    header button {
        display: none !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def get_mysql_engine():
    engine_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    engine = sqlalchemy.create_engine(engine_url)
    return engine

def fetch_table_list(engine):
    insp = sqlalchemy.inspect(engine)
    tables = insp.get_table_names()
    return tables

def main():
    """
    1. Setup page (hide menu and buttons)
    2. Connect to MySQL
    3. Select table
    4. Query the table with LLM
    """
    setup()
    
    # Display the logo at the top of the sidebar
    st.sidebar.image("greenlogo.png", use_container_width=True)
    st.sidebar.header("Select a table", divider="green")

    engine = get_mysql_engine()
    tables = fetch_table_list(engine)

    if not tables:
        st.write("No tables found in the database.")
        st.stop()

    selected_table = st.sidebar.selectbox("", options=tables)
    if not selected_table:
        st.stop()

    df = pd.read_sql_table(selected_table, con=engine)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Use the OpenAI LLM now
    llm = OpenAI(api_token="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")  # Or omit api_token if set in your environment
    connector = PandasConnector({"original_df": df})
    sdf = SmartDataframe(
        connector, 
        {"enable_cache": False}, 
        config={
            "llm": llm,
            "conversational": False,
            "response_parser": OutputParser
        }
    )

    prompt = st.text_input("Enter your question")
    if not prompt:
        st.stop()

    st.write("Response")
    response = sdf.chat(prompt)

if __name__ == '__main__':
    matplotlib.use("Agg", force=True)
    main()
