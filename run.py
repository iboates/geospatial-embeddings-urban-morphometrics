from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

from urban_morphometrics.main import compute_urban_morphometrics


def run_main():

    pbf_url = "https://download.geofabrik.de/north-america/us/washington-latest.osm.pbf"
    region_gdf = geocode_to_region_gdf("King County, Washington")
    regionalizer = H3Regionalizer(resolution=9)
    regions_gdf = regionalizer.transform(region_gdf)
    print(f"Number of cells: {len(regions_gdf)}")
    regions_gdf.to_file("urban_morphometrics/regions.gpkg")

    metric_config = {
        "knn_k": 15,
        "tessellation_buffer": 5,
        "tessellation_min_buffer": 0,
        "tessellation_max_buffer": 10,
        "tessellation_shrink": 0.4,
        "tessellation_segment": 0.5,
        "street_alignment_max_distance": 500.0,
    }

    compute_urban_morphometrics(
        study_area_gdf=regions_gdf,
        pbf_path=pbf_url,
        run_name="king_county_00",
        neighbourhood_distance=300,
        num_quantiles=4,
        equal_area_crs="EPSG:5070",
        equidistant_crs="EPSG:2926",
        conformal_crs="EPSG:2926",
        metric_config=metric_config,
        # metrics=["centroid_corner_distance"],
        debug=True,
        output_folder="urban_morphometrics_king_county",
        use_cache=True,
        export_features=True,
        n_workers=4,
    )


if __name__ == "__main__":
    run_main()
