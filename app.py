###############################################################################
# Source code of the application : GreenVidaFarms
# Authors: Andre Juarez, Marvin Quispe, Nicole Barrientos
###############################################################################

'''
MIT License

Copyright (c) 2025 - GreenVidaFarms

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

contact: greenvidafarms@gmail.com

'''

###############################################################################
# 1. LOAD LIBRARIES
###############################################################################

import os
import glob
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
from shiny import App, reactive, render, ui, Session
from shiny import App, reactive, render, ui, Session

# For graphics
import plotly.express as px
import plotly.graph_objects as go
import leafmap
from faicons import icon_svg
from modules.img import img_ui
from modules.img import img_server
from modules.train_forecast import forecast_prod
from leafmap.toolbar import change_basemap
from shiny.types import ImgData

# For forecasting and ai-chatbot
#import statsmodels.api as sm
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from shinywidgets import output_widget, render_widget
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from price_analysis_page import (
    price_forecast_for_categories,
    plot_price_vs_volume_with_monthly_line,
    category_groups,
    plot_category_group_with_dual_axes,calculate_relative_change_with_flag
)

###############################################################################
# 2. API's & LLM SETUP
###############################################################################

app_dir = Path(__file__).parent
#css_file = Path(__file__).parent / "www" / "css" / "styles.css"
#agent_output_path = Path(__file__).parent / "tmp" / "quarto_report" / "agent_output.qmd"
image_dir = os.path.join(app_dir, 'tmp', 'images')

# Set API's
openai_api_key = os.getenv("OPENAI_API_KEY", "")
tavily_api_key_file = os.getenv("TAVILY_API_KEY","")

if not openai_api_key or not tavily_api_key_file:
    raise ValueError("Please set both the OPENAI_API_KEY and TAVILY_API_KEY environment variables.")

# Create a GPT-4 chat model
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=openai_api_key,
    streaming=True
)

# Build an agent to answer questions about df
def get_image_files():
    return glob.glob(os.path.join(image_dir, '*.png'))

def get_image_files_len():
    return len(glob.glob(os.path.join(image_dir, '*.png')))

def reset_images_folder():
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    for image_file in image_files:
        try:
            os.remove(image_file)
        except Exception as e:
            print(f"Error deleting file {image_file}: {e}")

###############################################################################
# 2. UI
###############################################################################

page1_ui = ui.page_fluid(
    ui.card(
        ui.HTML(
            "<h3>Welcome!</h3>"
            "<p><em>Welcome to GreenVidaFarms' dashboard, your AI-driven platform dedicated to enhancing food security in Metropolitan Lima. This tool provides forecasts and real-time insights into daily market prices and volumes at 'Mercado Mayorista', as well as monthly earnings for farmers, empowering you to make informed decisions for a fair and sustainable food system.</em></p>"
        )),

    # Top row: Two charts side by side
    ui.layout_columns(
        ui.card(
            ui.card_header(ui.HTML(f"{icon_svg('ranking-star')} Top 5 Categories Forecast")),
            output_widget("top_5_forecast")
        ),
        ui.card(
            ui.card_header(ui.HTML(f"{icon_svg('chart-pie')} Bubble Chart: Volume vs. Price")),            
            output_widget("bubble_chart")
        ),
        col_widths=(8, 4)  # Specify column widths for the two charts
    ),

    # Middle row: Individual category selection
    ui.layout_columns(
        ui.card(
            ui.card_header(ui.HTML(f"{icon_svg('filter')} Select and Generate Category Graph")),
            ui.input_selectize(
                "category_select", 
                "Select Category:", 
                choices= [], 
                selected=None
            ),
            ui.input_select(
                "days_select", 
                "Select Number of Days:",
                choices={
                    30: "30 Days",
                    90: "90 Days",
                    180: "180 Days",
                    365: "1 Year",
                    1460: "4 Years"
                },
                selected=30  # Default to 30 days
            ),
            ui.input_action_button("generate_category_graph", "Generate Graph"),
            ui.output_ui("category_price_vs_volume")  # Price vs Volume graph
        ),
        col_widths=(12,)
    ),

    # Bottom row: Grouped bar chart
    ui.layout_columns(
        ui.card(
            ui.card_header(ui.HTML(f"{icon_svg('filter')} Select Category Group")),
            ui.input_select(
                "group_select", 
                "Select Category Group:", 
                choices=list(category_groups.keys()), 
                selected=list(category_groups.keys())[0]  # Default to the first group
            ),
            ui.input_select(
                "group_days_select", 
                "Select Number of Days:",
                choices={
                    30: "30 Days",
                    90: "90 Days",
                    180: "180 Days",
                    365: "1 Year",
                    1460: "4 Years"
                },
                selected=30  # Default to 30 days
            ),
            ui.input_action_button("generate_group_graph", "Generate Group Graph"),
            output_widget("group_bar_chart")  # Output widget for the grouped bar chart
        ),
        col_widths=(12,)
    )
)

page1 = ui.nav_panel("Price Analysis", page1_ui)

page2_ui = ui.page_fluid(
    ui.layout_columns(
        # Sidebar: Inspirational message, Reset Images button, and graph display
        ui.layout_columns(
            ui.card(
                ui.HTML(
                    "<h3>Welcome!</h3>"
                    "<p><em>Here is the GreenVidaFarms ChatBot. This assistant will help you answer any doubts concerning the daily price data on Lima Market's and give tailored recommendations for adquiring the best deals based on nutritional aspects and price data. Also, this ChatBot is trained to suggest the best recipes of the peruvian heritage based on this price recommendations as well as availabillity and nutritious value </em></p>"
                ),
            ),
            
            ui.card(
                ui.input_action_button('reset_button', 'Reset Images', style="margin-top: 10px;"),
                ui.card(
                    ui.card_header(ui.HTML(f"{icon_svg('images')} Plots Created")),
                    ui.layout_column_wrap(id_="images_container", width= 1),
                    ui.output_image("comidas_image"),
                    ui.output_image("mercado_image"),
                    full_screen=True
                    #max_height='50%'
                    #style="width: 100%;"
                )
            ),
            col_widths=(12,)
        ),
        # Main content: Chatbot UI
        ui.card(
            ui.card_header(ui.HTML(f"{icon_svg('comment')} GPT-4 Chatbot")),
            ui.card_body(
                ui.chat_ui(id="chat_area", placeholder="Ask me anything about the data...")
            ),
            full_screen=True,
            max_height='50%'
        ),
        col_widths=(4, 8),  # Sidebar is 4 columns wide, main content is 8 columns wide
        gap="20px",  # Add spacing between columns
        fillable=True
    )
)

page2 = ui.nav_panel("Chatbot", page2_ui)

page3_ui = ui.page_fluid(
    ui.layout_columns(
        ui.card(
            ui.card_header(ui.HTML(f"{icon_svg('map-location-dot')} Supermarkets and markets")),
            output_widget("map1")
        ),
        ui.card(
            ui.card_header(ui.HTML(f"{icon_svg('map-location-dot')} Communal pots / Ollas comunes")),
            output_widget("map2")
        ),
        col_widths=(12, )
    )
)

page3 = ui.nav_panel("Maps", page3_ui)

# The top-level UI with a navbar
app_ui = ui.page_navbar(
    page1,
    page2,
    page3,
    ui.nav_spacer(),
    ui.nav_control(ui.input_dark_mode(mode = 'light')),
    ui.nav_control(ui.HTML("<div id='google_translate_element'></div><script type='text/javascript'>function googleTranslateElementInit() {new google.translate.TranslateElement({pageLanguage: 'en'}, 'google_translate_element');}</script><script type='text/javascript' src='//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit'></script>")),
    title = "GreenVidaFarms App",
)


###############################################################################
# 3. SERVER
###############################################################################

message = {
        "content": "**Hello!** How can I help you today? - ¿Cómo puedo ayudarte?",
        "role": "assistant"
    }

user_messages = []

# Update data
import price_analysis_page  # We’ll see how to adjust it below
from modules.price_data_scrapping import update_price_data
from modules.volumen_data_scrapping import update_volume_data
from modules.train_forecast import forecast_prod
from modules.preproc_data import update_chatbot_data

def server(input, output, session: Session):
    ###############################################################################
    # (A) PRICE ANALYSIS (PAGE 1)
    ###############################################################################
    selected_images = reactive.Value({})
    category_list = reactive.Value([])

    # Reactive values to store data
    def data_update_merge_and_refresh_chatbot():
        # Scrape & forecast data if needed
        updated = update_price_data()# merges daily

        if updated:
            update_volume_data()
            forecast_prod()
            # Possibly also update volume data if you want
            
        # Then run the final merge to produce new chatbot XLSX files
        update_chatbot_data()  # from step #1
        # Rebuild data (for chatbot usage)
    
    data_update_merge_and_refresh_chatbot()

    aggregated_precio = pd.read_excel("data/data_raw/precio_final_merged_latest.xlsx")
    filtered_volumen  = pd.read_excel("data/data_raw/volumen_final_merged_latest.xlsx")
    df_forecast       = pd.read_excel("data/data_raw/forecast_latest.xlsx")
    precio_chakra     = pd.read_excel("data/data_raw/chakra_final_merged_v3.xlsx")
    
    # Convert date columns
    aggregated_precio["Extraction Date"] = pd.to_datetime(aggregated_precio["Extraction Date"])
    filtered_volumen["Extraction Date"]  = pd.to_datetime(filtered_volumen["Extraction Date"])
    df_forecast["ds"]                    = pd.to_datetime(df_forecast["ds"])
    categories = aggregated_precio['CATEGORIA'].unique()
    results = [
        calculate_relative_change_with_flag(aggregated_precio, category)
        for category in categories
    ]
    results = [res for res in results if res is not None]
    final_results_sorted = pd.DataFrame(results).sort_values(by='relative_change_30_15', ascending=True)
    top_5_categories = final_results_sorted['CATEGORIA'].head(5).tolist()
    
    # 20 categories
    #top_20_categories  = final_results_sorted['CATEGORIA'].head(20).tolist()

    # Build merged data for daily price/volume:
    price_volume_data = pd.merge(filtered_volumen, aggregated_precio, 
                                 on=["CATEGORIA", "Extraction Date"], how="outer")
    price_volume_data["precio_prom"] = price_volume_data["precio_prom"].fillna(method="ffill")
    price_volume_data["Volumen"]     = price_volume_data["Volumen"].fillna(0)
    lista_categories = price_volume_data['CATEGORIA'].unique().tolist()
    price_volume_data_grouped = price_volume_data.groupby(['CATEGORIA']).agg(
        precio_prom=('precio_prom', 'mean'),  # Average price
        Volumen=('Volumen', 'sum')           # Total volume
    ).reset_index() 
    precio_chakra['Date'] = pd.to_datetime(
            precio_chakra['AÑO'].astype(str) + '-' + precio_chakra['MES'].astype(str) + '-01'
        )
    
    # Now you can pass these DataFrames to your `price_analysis_page` module
    # so that the module uses the "live" copies, instead of reading from disk again. 
    price_analysis_page.set_data(
        aggregated_precio=aggregated_precio,
        filtered_volumen=filtered_volumen,
        df_forecast=df_forecast,
        precio_chakra=precio_chakra
    )
    category_list.set(lista_categories)
    
    def refresh_chatbot_data():
        """Loads the newly merged XLSX files, rebuilds the prefix, 
           and re-initializes the agent creation logic."""
        nonlocal PREFIX_CASE_STUDY, precio_volumen_mercado_diario, precio_agricultores_mensual, recetario_excel

        # Load new XLSX
        precio_volumen_mercado_diario_pre = pd.read_excel("data/data_chatbot/precio_volumen_mercado_diario.xlsx")
        precio_agricultores_mensual_pre   = pd.read_excel("data/data_chatbot/precio_agricultores_mensual.xlsx")
        recetario_excel                   = pd.read_excel("data/data_chatbot/recetario_excel.xlsx")

        #
        date_now = datetime.now()
        date_limit = date_now - timedelta(days=15)        
        precio_volumen_mercado_diario = precio_volumen_mercado_diario_pre[precio_volumen_mercado_diario_pre['Extraction Date'] >= date_limit]
        precio_agricultores_mensual   = precio_agricultores_mensual_pre[precio_agricultores_mensual_pre['Extraction Date'] >= date_limit]

        # Build new prefix
        filtered_df  = precio_volumen_mercado_diario[precio_volumen_mercado_diario['FORECAST_TYPE'].isnull()]
        last_hist_date = filtered_df['Extraction Date'].max()
        filtered_df2 = precio_agricultores_mensual[precio_agricultores_mensual['FORECAST_TYPE'].isnull()]
        last_date_chakra = filtered_df2['Extraction Date'].max()

        PREFIX_CASE_STUDY = f"""
            You are a pandas agent. You must work with the DataFrame 'precio_volumen_mercado_diario' containing daily price and volume data for the wholesale market in Lima, Peru.
            The DataFrame has the following columns: ['CATEGORIA', 'Extraction Date', 'precio_prom', 'Volumen', 'FORECAST_TYPE'].
            The user may ask you questions about specific categories, dates and prices.

            Follow these useful instructions when retrieving information:

            1. **Introduce**
            - If someone says hello or greets you for the first time, tell them you are an virtual assistant called **GreenVidaFarms** and how you can help them and what data you have access to.

            2. **Upload dataframe**
            -   Use this code to import dataframe before any analysis (The analysis includes the creation of graphs, obtaining minimums/maximums, unique categories, etc. anything that is related to the dataframe):
                    precio_volumen_mercado_diario = pd.read_excel("data/data_chatbot/precio_volumen_mercado_diario.xlsx")

            3. **Unique Categories**: 
            - If the user asks for unique categories, use the `.unique()` method on the 'CATEGORIA' column to retrieve all unique rows in natural language.

            4. **Plotting Data**:
            - If the user asks for a plot, generate a line plot showing 'precio_prom' over time for the specified 'CATEGORIA'. Use the following steps:
                - Use 'Extraction Date' for the x-axis and 'precio_prom' for the y-axis.
                - Ensure the plot has clear axis labels, a title mentioning the category, and appropriate formatting.
                - If no category is specified, notify the user that a category is required for plotting.
                - Additionally, Always save the plots to a temporary folder using: os.path.join(os.getcwd(), 'tmp/images').
            
            5. **Recipe book**
            - If someone asks you for a recipe the first thing to do is to consult the recipe database which you have to load by executing the following code: 
                recetario_excel = pd.read_excel("data/data_chatbot/recetario_excel.xlsx")
            - To better understand the cookbook first calculate a df.head() to know its columns and rows
 
        """
    precio_volumen_mercado_diario = None
    precio_agricultores_mensual   = None
    recetario_excel               = None
    PREFIX_CASE_STUDY = ""
    refresh_chatbot_data()

    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)
    tools = [tavily_tool]

    def make_agent():
        return create_pandas_dataframe_agent(
            llm=model,
            df=precio_volumen_mercado_diario,
            include_df_in_prompt = True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=PREFIX_CASE_STUDY,
            allow_dangerous_code=True,
            verbose=True, 
            extra_tools = tools)

    @reactive.Effect
    def update_category_ui():
        # The new categories we want to present
        lista_categorias = category_list.get()
        # The user’s current selection
        current_value = input.category_select()
        print("Current selection before update:", current_value)

        # Call the dynamic update function from shinywidgets
        ui.update_selectize(
            "category_select",
            choices=lista_categorias,
            selected=current_value
        )
        print("Current selection after update:", input.category_select())

    @render_widget
    def bubble_chart():
        print("Bubble Chart Data Preview:", price_volume_data_grouped.head())

        fig = px.scatter(
        price_volume_data_grouped,
        x='Volumen',                # X-axis: volume
        y='precio_prom',            # Y-axis: average price
        size='Volumen',             # Bubble size: volume
        color='CATEGORIA',          # Bubble color by category
        hover_name='CATEGORIA',     # Hover info: category
        title="Relación entre Volúmenes y Precios Promedio",
        labels={'Volumen': 'Volumen (Producción)', 'precio_prom': 'Precio Promedio'}
        )
        return fig
    
    @render_widget
    def top_5_forecast():
        fig = price_forecast_for_categories(top_5_categories, price_volume_data,df_forecast, display_days=30)
        return fig

    @output
    @render.ui
    def category_price_vs_volume():
        # Wait for the button to be pressed
        input.generate_category_graph()

        # Get the selected category and number of days
        categoria = input.category_select()
        num_days = int(input.days_select())  # Convert the selected number of days to an integer

        # Generate the plot with the selected number of days
        fig = plot_price_vs_volume_with_monthly_line(price_volume_data, precio_chakra, df_forecast,category=categoria, num_days=num_days)

        # Render the plot
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    @render_widget
    def group_bar_chart():
        # Wait for the button to be pressed
        input.generate_group_graph()

        # Get the selected group and number of days
        group_name = input.group_select()
        num_days = int(input.group_days_select())  # Convert the selected number of days to an integer

        # Generate the grouped bar plot
        fig = plot_category_group_with_dual_axes(price_volume_data, category_groups, group_name, days=num_days)
        
    # Render the plot
        return fig    
    chat = ui.Chat(
        id="chat_area",
        messages=[message])

    @chat.on_user_submit
    async def _():
        """
        Runs whenever the user presses Enter/Send in the chat.
        We'll pass the user's text to GPT-4, then show the answer.
        """
        messages = chat.messages(format='langchain', token_limits=None)
        user_message = chat.user_input()
        user_messages.append(user_message)

        #Append user message (so user sees it in the conversation)
        #await chat.append_message({"role": "user", "content": user_message})

        #Call the agent
        agent = make_agent()

        try:
            ans_dict = agent.invoke(messages)
            answer = ans_dict["output"]

        except Exception as e:
            answer = f"Error: {e}"

        # Append the assistant's answer
        # await chat.append_message_stream({"role": "assistant", "content": answer})
        await chat.append_message_stream(answer)

    @reactive.poll(get_image_files_len, 0.5)
    def get_image():
        return get_image_files()

    current_files = reactive.Value(get_image_files())

    for image_file in get_image_files():
        image_id = os.path.basename(image_file).replace(".", "_")
        ui.insert_ui(
            selector="#images_container",
            ui=ui.TagList(img_ui(image_id)),
            where="beforeEnd"
        )
        img_server(id=image_id, file=image_file, selected_images=selected_images)

    @reactive.Effect
    @reactive.event(input.reset_button)
    def reset_images():
        reset_images_folder()  # Call the function to delete images
        selected_images.set({})  # Clear the selected images dictionary
        ui.remove_ui(selector="#images_container")

    @reactive.Effect
    @reactive.event(get_image)
    def update_images():
        current_images = set(os.path.basename(f) for f in current_files.get())
        new_images = set(os.path.basename(f) for f in get_image())

        # Detect newly added images
        images_to_add = new_images - current_images

        # Add new images to the UI
        for i, image_file in enumerate(images_to_add):
            full_path = os.path.join(image_dir, image_file)
            image_id = image_file.replace(".", "_") + f"_{i}"
            ui.insert_ui(
                selector="#images_container",
                ui=img_ui(image_id),
                where="beforeEnd"
            )
            img_server(id=image_id, file=full_path, selected_images=selected_images)
    
    www_dir = Path(__file__).parent / "www/img"

    @output
    @render.image
    def comidas_image() -> ImgData:
        return {
            "src": str(www_dir / "comidas.png"),
            "width": "100%",  # Adjust as needed
            "alt": "Comidas Image"
        }

    @output
    @render.image
    def mercado_image() -> ImgData:
        return {
            "src": str(www_dir / "mercado.png"),
            "width": "100%",  # Adjust as needed
            "alt": "Mercado Image"
        }

    markets = pd.read_excel("data/data_maps/MARKETS_DATA_2016.xlsx")

    def create_custom_map1():
        geo_df = gpd.GeoDataFrame(markets, geometry = gpd.points_from_xy(markets.LONGITUD, markets.LATITUD), crs="EPSG:4326")
        fig = px.scatter_map(geo_df,
                        lat=geo_df.geometry.y,
                        lon=geo_df.geometry.x,
                        hover_name="NOMBRE",
                        zoom=10)

        return fig
    
    def load_ollas_data():
        # Load the data
        ollas = pd.read_excel("data/data_maps/OLLAS_DE_LIMA.xlsx")

        # Replace commas with dots in latitude and longitude
        ollas['Latitude'] = ollas['Latitude'].astype(str).str.replace(',', '.')
        ollas['Longitude'] = ollas['Longitude'].astype(str).str.replace(',', '.')

        # Convert to numeric, setting errors='coerce' to handle invalid entries
        ollas['Latitude'] = pd.to_numeric(ollas['Latitude'], errors='coerce')
        ollas['Longitude'] = pd.to_numeric(ollas['Longitude'], errors='coerce')

        # Drop rows with NaN values in 'Latitude' or 'Longitude'
        ollas = ollas.dropna(subset=['Latitude', 'Longitude'])

        # Optional: Remove entries with invalid coordinate ranges
        ollas = ollas[
            (ollas['Latitude'] >= -90) & (ollas['Latitude'] <= 90) &
            (ollas['Longitude'] >= -180) & (ollas['Longitude'] <= 180)
        ]

        return ollas

    data_ollas = load_ollas_data()
    data_ollas = data_ollas.reset_index(drop=True)

    def create_custom_map2():
        geo_df = gpd.GeoDataFrame(data_ollas, geometry = gpd.points_from_xy(data_ollas.Longitude, data_ollas.Latitude), crs="EPSG:4326")
        fig = px.scatter_map(geo_df,
                        lat=geo_df.geometry.y,
                        lon=geo_df.geometry.x,
                        hover_name="NOMBRE",
                        zoom=10)

        return fig

    @render_widget
    def map1():
        # Llamar a la función para crear el mapa personalizado
        return create_custom_map1()

    @render_widget
    def map2():
        # Llamar a la función para crear el mapa personalizado
        return create_custom_map2()

###############################################################################
# 3. CREATE AND RUN THE APP
###############################################################################

app = App(app_ui, server,static_assets=app_dir / "www")