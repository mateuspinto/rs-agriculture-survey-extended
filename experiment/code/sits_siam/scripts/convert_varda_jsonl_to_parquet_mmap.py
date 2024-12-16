import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import shape, MultiPolygon
from tqdm.std import tqdm


def read_jsonl_to_geodataframe(file_path, n_rows=None, buffer_size=100 * 1024 * 1024):
    """
    Reads a JSONL file incrementally with buffered reading and builds a GeoDataFrame.

    Args:
    - file_path (str): Path to the JSONL file.
    - n_rows (int, optional): Number of rows to read for testing purposes.
    - buffer_size (int, optional): Size of the buffer in bytes for reading the file.

    Returns:
    - gdf (GeoDataFrame): The resulting GeoDataFrame.
    """
    # Count the total number of lines using buffered reading
    # total_lines = 0
    # with open(file_path, "rb") as file:
    #     while chunk := file.read(buffer_size):
    #         total_lines += chunk.count(b"\n")

    # Initialize an empty GeoDataFrame
    gdf = None

    # Open the file and process it with tqdm
    with open(file_path, "r") as file, tqdm(
        total=n_rows,  # min(total_lines, n_rows) if n_rows else total_lines,
        desc="Processing lines",
    ) as pbar:
        buffer = ""
        line_count = 0

        while True:
            chunk = file.read(buffer_size)
            if not chunk:
                break

            buffer += chunk
            lines = buffer.splitlines()

            # Keep the last line (might be incomplete)
            buffer = lines.pop() if chunk else ""

            for line in lines:
                # Stop reading if the limit of rows is reached
                if n_rows and line_count >= n_rows:
                    break

                # Load the current JSON line
                row = json.loads(line)

                # Extract geometry and cast MultiPolygon to Polygon if applicable
                geometry = shape(row["geometry"])
                if isinstance(geometry, MultiPolygon) and len(geometry.geoms) == 1:
                    geometry = geometry.geoms[0]  # Cast to Polygon

                # Extract the area value and convert to hectares
                area_value = (
                    row.get("properties", {}).get("area", {}).get("value") / 10000
                )

                # Only process geometries larger than 6.25 ha
                if area_value > 6.25:
                    # Convert the ID to string
                    feature_id = row.get("id")

                    # Extract other known columns
                    # bbox = row.get("bbox")
                    # properties = row.get("properties", {})
                    # area_unit = properties.get("area", {}).get("unit")
                    # boundary_references = properties.get("boundary_references")
                    # centroid = properties.get("centroid")
                    # country_iso_codes = properties.get("country_iso_codes")
                    # field_relationships = properties.get("field_relationships")
                    # perimeter_unit = properties.get("perimeter", {}).get("unit")
                    # perimeter_value = properties.get("perimeter", {}).get("value")
                    # representative_point = properties.get("representative_point")
                    # feature_type = row.get("type")

                    # Create a DataFrame for the current row
                    row_df = gpd.GeoDataFrame(
                        {
                            "geometry": [geometry],
                            # "bbox": [bbox],
                            "id": [feature_id],
                            # "area_unit": [area_unit],
                            "ha": [area_value],
                            # "boundary_references": [boundary_references],
                            # "centroid": [centroid],
                            # "country_iso_codes": [country_iso_codes],
                            # "field_relationships": [field_relationships],
                            # "perimeter_unit": [perimeter_unit],
                            # "perimeter_value": [perimeter_value],
                            # "representative_point": [representative_point],
                            # "feature_type": [feature_type],
                        },
                        geometry="geometry",
                        crs="EPSG:4326",
                    )

                    # Initialize the GeoDataFrame or append to it
                    if gdf is None:
                        gdf = row_df
                    else:
                        gdf = pd.concat([gdf, row_df], ignore_index=True)

                # Update counters
                line_count += 1
                pbar.update(1)

    return gdf


def save_geodataframe_to_parquet(gdf, output_path):
    """
    Saves a GeoDataFrame to a Parquet file with Brotli compression.

    Args:
    - gdf (GeoDataFrame): The GeoDataFrame to save.
    - output_path (str): Path to the output Parquet file.
    """
    gdf.to_parquet(output_path, compression="brotli")


# Example usage
file_path = (
    "data/varda-fieldid-2024-07-USA.geojsonl"  # Replace with your JSONL file path
)
output_path = "data/varda-fieldid-2024-07-USA.parquet"  # Replace with your desired output file path
n_rows = 1_000_000  # Set to None to process the entire file

# n_rows = 28_896_457 # USA
# Read JSONL to GeoDataFrame
gdf = read_jsonl_to_geodataframe(file_path, n_rows=n_rows)

# Save GeoDataFrame to Parquet
if gdf is not None:
    save_geodataframe_to_parquet(gdf, output_path)
    print(f"GeoDataFrame saved to {output_path}")
else:
    print("No geometries larger than 6.25 ha were found.")
