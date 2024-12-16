import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import shape, MultiPolygon
from tqdm.std import tqdm
import os


def count_lines(file_path):
    """
    Counts the number of lines in a file using the 'wc -l' command (Linux/Mac).
    """
    result = os.popen(f"wc -l {file_path}").read()
    return int(result.split()[0])


def read_jsonl_to_geodataframe(file_path, n_rows=None):
    """
    Reads a JSONL file and incrementally builds a GeoDataFrame.

    Args:
    - file_path (str): Path to the JSONL file.
    - n_rows (int, optional): Number of rows to read for testing purposes.

    Returns:
    - gdf (GeoDataFrame): The resulting GeoDataFrame.
    """
    # Get total number of lines in the file
    total_lines = count_lines(file_path)

    # Initialize an empty GeoDataFrame
    gdf = None

    with open(file_path, "r") as file:
        for i, line in enumerate(
            tqdm(
                file,
                total=min(total_lines, n_rows) if n_rows else total_lines,
                desc="Processing lines",
            )
        ):
            # Stop reading if the limit of rows is reached
            if n_rows and i >= n_rows:
                break

            # Load the current JSON line
            row = json.loads(line)

            # Extract geometry and cast MultiPolygon to Polygon if applicable
            geometry = shape(row["geometry"])
            if isinstance(geometry, MultiPolygon) and len(geometry.geoms) == 1:
                geometry = geometry.geoms[0]  # Cast to Polygon

            feature_id = row.get("id")

            # Extract other known columns
            # bbox = row.get("bbox")
            properties = row.get("properties", {})
            # area_unit = properties.get("area", {}).get("unit")
            area_value = properties.get("area", {}).get("value")
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
                    "area_value": [area_value],
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
    "data/varda-fieldid-2024-10-BRA.geojsonl"  # Replace with your JSONL file path
)
output_path = "data/varda-fieldid-2024-10-BRA.parquet"  # Replace with your desired output file path
n_rows = 100  # Set to None to process the entire file

# Read JSONL to GeoDataFrame
gdf = read_jsonl_to_geodataframe(file_path, n_rows=n_rows)

# Save GeoDataFrame to Parquet
save_geodataframe_to_parquet(gdf, output_path)

print(f"GeoDataFrame saved to {output_path}")
