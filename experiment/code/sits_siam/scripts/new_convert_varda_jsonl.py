import geopandas as gpd
import pandas as pd
import json
import gc
from shapely.geometry import shape, MultiPolygon
from tqdm import tqdm


def save_geodataframe_to_parquet(gdf, output_path):
    """
    Saves a GeoDataFrame to a Parquet file with Brotli compression.

    Args:
    - gdf (GeoDataFrame): The GeoDataFrame to save.
    - output_path (str): Path to the output Parquet file.
    """
    gdf.to_parquet(output_path, compression="brotli")


def process_jsonl_in_batches(
    file_path,
    batch_size=1_000_000,
    buffer_size=100 * 1024 * 1024,
    output_prefix="output",
):
    """
    Reads a JSONL file incrementally with buffered reading, processes data in batches,
    and saves each batch to a Parquet file.

    Args:
    - file_path (str): Path to the JSONL file.
    - batch_size (int, optional): Number of lines per batch to process and save.
    - buffer_size (int, optional): Size of the buffer in bytes for reading the file.
    - output_prefix (str, optional): Prefix for the output Parquet files.
    """

    # Initialize variables
    batch_data = []
    batch_count = 0
    line_count = 0

    # Open the file and process it with tqdm
    with open(file_path, "r") as file, tqdm(
        desc="Processing lines",
    ) as pbar:
        buffer = ""

        while True:
            chunk = file.read(buffer_size)
            if not chunk:
                # Process any remaining buffer
                if buffer:
                    lines = [buffer]
                    buffer = ""
                else:
                    break
            else:
                buffer += chunk
                lines = buffer.splitlines()
                buffer = lines.pop() if not chunk.endswith("\n") else ""

            for line in lines:
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
                    feature_id = str(row.get("id"))

                    # Create a dict for the current row
                    row_data = {
                        "geometry": geometry,
                        "id": feature_id,
                        "ha": area_value,
                    }

                    # Append to batch data
                    batch_data.append(row_data)

                # Update counters
                line_count += 1
                pbar.update(1)

                # When batch size is reached, process and save
                if line_count % batch_size == 0:
                    if batch_data:
                        # Create GeoDataFrame
                        gdf = gpd.GeoDataFrame(
                            batch_data,
                            geometry="geometry",
                            crs="EPSG:4326",
                        )

                        # Save to Parquet
                        output_path = f"{output_prefix}_{batch_count}.parquet"
                        save_geodataframe_to_parquet(gdf, output_path)
                        print(f"Batch {batch_count} saved to {output_path}")

                        # Free memory
                        del gdf
                        batch_data = []
                        gc.collect()
                    else:
                        print(f"No data to save in batch {batch_count}")

                    batch_count += 1

        # After finishing reading, process remaining data
        if batch_data:
            gdf = gpd.GeoDataFrame(
                batch_data,
                geometry="geometry",
                crs="EPSG:4326",
            )
            output_path = f"{output_prefix}_{batch_count}.parquet"
            save_geodataframe_to_parquet(gdf, output_path)
            print(f"Batch {batch_count} saved to {output_path}")

            # Free memory
            del gdf
            batch_data = []
            batch_count += 1
            gc.collect()


# Example usage
file_path = (
    "data/varda-fieldid-2024-07-USA.geojsonl"  # Replace with your JSONL file path
)
output_prefix = (
    "data/varda-fieldid-2024-07-USA"  # Replace with your desired output file prefix
)
batch_size = 1_000_000  # Number of lines per batch

# Process JSONL in batches and save to Parquet
process_jsonl_in_batches(file_path, batch_size=batch_size, output_prefix=output_prefix)
