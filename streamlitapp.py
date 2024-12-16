import streamlit as st
import pandas as pd
import sqlalchemy
from pandasai import SmartDataframe
from pandasai.connectors import PandasConnector
from pandasai.llm import BambooLLM
from pandasai.responses.response_parser import ResponseParser
import os
import matplotlib

# Set your PandasAI API Key
os.environ['PANDASAI_API_KEY'] = '$2a$10$zYXEMaIw0fLSjGSq3MrVDuNV9HQ85BfV4O5s5fxSBoCrediuGQcg2'

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
    st.header("Chat with your MySQL Tables!", anchor=False, divider="green")
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def get_mysql_engine():
    # Create a SQLAlchemy engine with given credentials
    engine_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    engine = sqlalchemy.create_engine(engine_url)
    return engine

def fetch_table_list(engine):
    # Fetch a list of all tables from the database
    insp = sqlalchemy.inspect(engine)
    tables = insp.get_table_names()
    return tables

def main():
    """
    1. Setup page
    2. Connect to MySQL
    3. Select table
    4. Provide optional column descriptors
    5. Query the table with LLM
    """
    setup()
    st.sidebar.header("Select a table from MySQL", divider="green")

    engine = get_mysql_engine()
    tables = fetch_table_list(engine)

    if not tables:
        st.write("No tables found in the database.")
        st.stop()

    selected_table = st.sidebar.selectbox("Choose a table:", options=tables)

    if not selected_table:
        st.stop()

    # Load the selected table into a DataFrame
    df = pd.read_sql_table(selected_table, con=engine)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Ask if user wants to add column descriptions
    col_desc = st.radio("Do you want to provide column descriptors?",
                        ("Yes", "No")
                       )
    if col_desc == "Yes":
        addon = st.text_input("Enter your column description, e.g. 'col1': 'unique id'")
    else:
        addon = "None"

    # Once descriptions are provided (or not), prepare LLM & PandasAI
    if addon:
        llm = BambooLLM()
        connector = PandasConnector({"original_df": df}, field_descriptions=addon)
        sdf = SmartDataframe(connector, {"enable_cache": False}, config={
            "llm": llm,
            "conversational": False,
            "response_parser": OutputParser
        })

        prompt = st.text_input("Enter your question/prompt.")
        if not prompt:
            st.stop()

        st.write("Response")
        response = sdf.chat(prompt)
        st.divider()
        st.write("🧞‍♂️ Under the hood, the code that was executed:")
        st.code(sdf.last_code_executed)

if __name__ == '__main__':
    # Ensure no interactive matplotlib backend issues
    matplotlib.use("Agg", force=True)
    main()
