# scripts/daily_update.py
import pandas as pd
import re
from datetime import datetime, timedelta

###############################################################################
# 1. LOAD EXISTING "aggregated_precio" DATA
###############################################################################
# We’ll define a "base" path for your existing data.
# If you always want to use one "latest" version, you can keep it at "precio_final_merged.xlsx".
# Or you can do versioning with a date suffix. Adjust as needed.
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def update_price_data() -> bool:
    BASE_DATA_PATH = "data/data_raw"
    BASE_FILENAME = "precio_final_merged_latest.xlsx"
    AGGREGATED_PRECIOS_FILE = f"{BASE_DATA_PATH}/{BASE_FILENAME}"

    aggregated_precio = pd.read_excel(AGGREGATED_PRECIOS_FILE)
    aggregated_precio['Extraction Date'] = pd.to_datetime(aggregated_precio['Extraction Date'], errors='coerce')
    last_date_price = aggregated_precio['Extraction Date'].max()
    print(f"Last aggregated date found: {last_date_price.date()}")

    start_date = last_date_price + timedelta(days=1)
    end_date   = datetime.now()

    print(f"Will look for new data from {start_date.date()} to {end_date.date()}")
    if start_date > end_date:
        print("No new data range found. Exiting.")
        return False

    monthly_data = pd.DataFrame()

    # Loop over each date from start_date to end_date
    current_date = start_date
    while current_date < end_date:
        # Convert current date to dd/mm/YYYY format
        fecha_str = current_date.strftime("%d/%m/%Y")

        reqUrl = "https://old.emmsa.com.pe/emmsa_spv/app/reportes/ajax/rpt07_gettable_new_web.php"
        headersList = {
            "Accept": "*/*",
            "User-Agent": "Thunder Client (https://www.thunderclient.com)",
            "Content-Type": "multipart/form-data; boundary=kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A"
        }

        # Insert fecha_str using f-string
        payload = (
            "--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A\r\n"
            "Content-Disposition: form-data; name=\"vid_tipo\"\r\n\r\n1\r\n"
            "--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A\r\n"
            "Content-Disposition: form-data; name=\"vfecha\"\r\n\r\n"
            f"{fecha_str}\r\n"
            "--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A--\r\n"
        )

        response = requests.request("POST", reqUrl, data=payload, headers=headersList)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"class": "timecard"})

            if table:
                data_rows = []
                for row in table.find("tbody").find_all("tr"):
                    columns = row.find_all("td")
                    data_rows.append([col.text.strip() for col in columns])

                column_names = ["Producto", "Variedad", "Precio min", "Precio max", "precio prom"]
                df = pd.DataFrame(data_rows, columns=column_names)
                # Mark the date scraped
                df["Extraction Date"] = fecha_str

                monthly_data = pd.concat([monthly_data, df], ignore_index=True)
                print(f"Data for {fecha_str} extracted successfully.")
            else:
                print(f"No table found for {fecha_str}.")
        else:
            print(f"Request failed with status {response.status_code} for {fecha_str}")

        current_date += timedelta(days=1)



    # ---------------
    # YOUR SELENIUM SCRAPER LOGIC HERE:
    # Example:
    # mayorista_precio = scrape_emmsa_prices(start_date, end_date)
    # ---------------
    # Or maybe you have CSV files for each date. We’ll just simulate it:

    # Convert 'Extraction Date' to datetime
    mayorista_precio = monthly_data.copy()

    mayorista_precio["Extraction Date"] = pd.to_datetime(
        mayorista_precio["Extraction Date"], format='%d/%m/%Y', errors='coerce', dayfirst=True
    )

    # Normalize to remove any time component
    mayorista_precio["Extraction Date"] = mayorista_precio["Extraction Date"].dt.normalize()
    ###############################################################################
    # 4. CATEGORY GROUPING LOGIC
    ###############################################################################
    def agrupar_producto(producto: str) -> str:
        """
        Groups the product name into a category based on the first
        word found before delimiters like '/', '('.
        """
        producto_normalizado = producto.upper().strip()
        tokens = re.split(r'[ /\(]', producto_normalizado)
        primera_palabra = next((token for token in tokens if token), producto_normalizado)
        return primera_palabra

    # Apply the 'agrupar_producto' function
    mayorista_precio["CATEGORIA"] = mayorista_precio["Variedad"].apply(agrupar_producto)

    ###############################################################################
    # 5. LOAD & DEFINE DESIRED CATEGORIES
    ###############################################################################
    LISTA_CATEGORIAS = pd.read_excel(f"{BASE_DATA_PATH}/merged_categoria_hortalizas.xlsx")

    # Suppose we do something like:
    # categories_all_present = LISTA_CATEGORIAS[LISTA_CATEGORIAS['PRESENCE_COUNT'] == 3]['CATEGORIA']
    # additional_categories = ['CACAO','CAFE','CHOCLO','COCO','ARROZ',
    #                         'DURAZNO','MANZANILLA','PIÑA','ROMERO','VAINITA']
    # final_categories = set(categories_all_present).union(set(additional_categories))

    # For illustration, we’ll do something simpler:
    categories_all_present = LISTA_CATEGORIAS[LISTA_CATEGORIAS['PRESENCE_COUNT'] == 3]['CATEGORIA']
    additional_categories = [
        'CACAO','CAFE','CHOCLO','COCO','ARROZ',
        'DURAZNO','MANZANILLA','PIÑA','ROMERO','VAINITA'
    ]
    final_categories = set(categories_all_present).union(set(additional_categories))

    selector_df = pd.DataFrame({'CATEGORIA': list(final_categories)})

    ###############################################################################
    # 6. FILTER NEW DATA FOR THESE CATEGORIES
    ###############################################################################
    filtered_precio = mayorista_precio[
        mayorista_precio['CATEGORIA'].isin(selector_df['CATEGORIA'])
    ].copy()
    for col in ['Precio min', 'Precio max', 'precio prom']:
        filtered_precio[col] = pd.to_numeric(filtered_precio[col])
    filtered_precio2 = filtered_precio.groupby(['CATEGORIA', 'Extraction Date']).agg(
        Precio_min=('Precio min', 'min'),  # Minimum price for the day
        Precio_max=('Precio max', 'max'),  # Maximum price for the day
        precio_prom=('precio prom', 'mean')).reset_index()
    ###############################################################################
    # 7. CONCATENATE THE OLD AND THE NEW
    ###############################################################################
    data_final = pd.concat([aggregated_precio, filtered_precio2], ignore_index=True)

    # Optionally drop duplicates


    print(f"Data shape after merging = {data_final.shape}")

    ###############################################################################
    # 8. SAVE OUT THE NEW "aggregated_precio" WITH A DATE-STAMP
    ###############################################################################
    # Example: precio_final_merged_YYYY_MM_DD.xlsx

    # If you also want to overwrite a “latest” version so your app always finds it:
    data_final.to_excel(f"{BASE_DATA_PATH}/precio_final_merged_latest.xlsx", index=False)
    print(data_final)
    print("Also updated 'precio_final_merged_latest.xlsx'")
    return True
###############################################################################
# DONE.
###############################################################################
update_price_data()