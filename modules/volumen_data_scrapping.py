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
def update_volume_data() -> bool:
    import pandas as pd
    import re
    from datetime import datetime, timedelta
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import re
    BASE_DATA_PATH = "data/data_raw"
    BASE_FILENAME = "volumen_final_merged_latest.xlsx"
    AGGREGATED_PRECIOS_FILE = f"{BASE_DATA_PATH}/{BASE_FILENAME}"

    # Attempt to load the existing aggregated file
    aggregated_precio = pd.read_excel(AGGREGATED_PRECIOS_FILE)

    # Ensure 'Extraction Date' is datetime
    aggregated_precio['Extraction Date'] = pd.to_datetime(aggregated_precio['Extraction Date'], errors='coerce')

    # Find the latest date we have in the existing file
    last_date_price = aggregated_precio['Extraction Date'].max()
    print(f"Last aggregated date found: {last_date_price.date()}")

    ###############################################################################
    # 2. DEFINE START & END DATES FOR SCRAPING OR CSV LOADING
    ###############################################################################
    # start_date is the day after 'last_date_price'
    start_date = last_date_price + timedelta(days=1)

    # end_date is "yesterday"
    end_date = datetime.now() 

    print(f"Will look for new data from {start_date.date()} to {end_date.date()}")

    # If the start_date > end_date, it means there’s no new data to add.
    if start_date >= end_date:
        print("[INFO] No new data range. Not updating.")
        return False




    # Initialize variables
    current_date = start_date
    monthly_data = pd.DataFrame()  # DataFrame to store data for each month

    # Process the date range
    while current_date < end_date:
        # Format the current date as a string
        fecha_str = current_date.strftime("%d/%m/%Y")

        import requests

        reqUrl = "https://old.emmsa.com.pe/emmsa_spv/app/reportes/ajax/rpt07_gettable_new_web.php"

        headersList = {
        "Accept": "*/*",
        "Content-Type": "multipart/form-data; boundary=kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A" 
        }

        payload = (
            "--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A\r\n"
            "Content-Disposition: form-data; name=\"vid_tipo\"\r\n\r\n2\r\n"
            "--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A\r\n"
            "Content-Disposition: form-data; name=\"vfecha\"\r\n\r\n"
            f"{fecha_str}\r\n"
            "--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A--\r\n"
        )


        response = requests.request("POST", reqUrl, data=payload,  headers=headersList)

        # Move to the next date
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "tbReport"})

            if table:
                data_rows = []
                for row in table.find("tbody").find_all("tr"):
                    columns = row.find_all("td")
                    data_rows.append([col.text.strip() for col in columns])

                column_names = ["Producto", "Variedad", "Volumen (TM)"]

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
    volumen_precio = monthly_data.copy()
    print(volumen_precio)
    volumen_precio["Extraction Date"] = pd.to_datetime(
        volumen_precio["Extraction Date"], format='%d/%m/%Y', errors='coerce'
    )
    volumen_precio["Extraction Date"] = volumen_precio["Extraction Date"].dt.strftime('%Y-%m-%d')
    volumen_precio["Extraction Date"] = pd.to_datetime(volumen_precio["Extraction Date"], format='%Y-%m-%d')

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
    volumen_precio["CATEGORIA"] = volumen_precio["Variedad"].apply(agrupar_producto)

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
    filtered_precio = volumen_precio[
        volumen_precio['CATEGORIA'].isin(selector_df['CATEGORIA'])
    ].copy()
    for col in ['Volumen (TM)']:
        filtered_precio[col] = pd.to_numeric(filtered_precio[col])
    filtered_precio2 = filtered_precio.groupby(['CATEGORIA', 'Extraction Date']).agg(
        Volumen=('Volumen (TM)', 'sum') # Minimum price for the day
    ).reset_index()
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
    data_final.to_excel(f"{BASE_DATA_PATH}/volumen_final_merged_latest.xlsx", index=False)
    print("Also updated 'volumen_final_merged_latest.xlsx'")
    return True
###############################################################################
# DONE.
###############################################################################
update_volume_data()