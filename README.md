# OSM-Constrained, Satellite-Evidence-Based Route Resilience Analysis

## 1. Architecture

This pipeline treats the OSM road network as the structural topology and the V5 Sentinel-2 U-Net probability raster as remote-sensing evidence. The design explicitly avoids building the resilience graph from a fragmented semantic-skeleton result.

The flow is:

1. Sentinel-2 AOI and V5 probability raster
2. OSM road centerlines
3. Per-road probability sampling along the centerline plus a small perpendicular corridor
4. Edge-level confidence aggregation and status labeling
5. OSM graph construction using road intersections and endpoints as nodes
6. Criticality analysis using degree, betweenness, bridge, and articulation metrics
7. Failure simulation on critical roads
8. Resilience score calculation and visualization

This is intentionally described as:

"OSM-constrained, satellite-evidence-based route resilience analysis."

## 2. Data flow

### Inputs
- Sentinel-2 AOI raster: `sentinel_bengaluru_large_aoi.tif`
- V5 probability raster: `bengaluru_v5_probability.tif`
- OSM roads: `bengaluru_large_roads_utm.gpkg`
- Trained model weights: `road_unet_v5_best.pth`

### Outputs
- `road_edge_confidence.gpkg`
- `bengaluru_road_graph.gpickle`
- `critical_roads.csv`
- `critical_nodes.csv`
- `resilience_results.csv`
- `top_critical_roads.gpkg`
- `visualizations/...`

## 3. Why not build the graph from the V5 skeleton?

The earlier skeleton-based graph was extremely fragmented because binary road segmentation forms a noisy, disconnected skeleton rather than a network topology. The project requirement is explicit: the V5 segmentation is probabilistic evidence, not the primary topology. The OSM network is the authoritative backbone structure and the model provides confidence support for each OSM edge.

## 4. Sampling strategy and corridor method

Each OSM line is sampled at regular intervals along its length. A robust implementation is used:

- centerline sample points every 5 m
- perpendicular sampling offsets of 0 m, 3 m, and 6 m
- mirrored side sampling at ±3 m and ±6 m to approximate a small corridor

This detects road confidence around the centerline without simply trusting one pixel. It is more stable than using only one central pixel, especially in satellite imagery where road masks may be slightly offset or jagged.

We choose the corridor-based method because roads are represented as centerlines in OSM, while the semantic model produces a road area or probability footprint. The corridor captures the region where the road is likely present rather than only the exact centerline location.

## 5. Confidence model

The confidence score is computed as a weighted heuristic in [0, 1]:

confidence = 0.35 * mean_prob + 0.25 * median_prob + 0.20 * frac_above_0.70 + 0.10 * continuity + 0.10 * support

where:
- mean_prob is the average sampled probability along the road corridor
- median_prob reduces sensitivity to outliers
- frac_above_0.70 is the proportion of samples in the high-confidence region
- continuity measures how long the road stays above the 0.55 threshold in contiguous runs
- support is a stable, bounded evidence term

This avoids a naive mean-only score, which can be skewed by sparse high-probability or noisy samples.

## 6. Status classification

The initial thresholds are heuristic and should be validated against ground-truth road condition data:

- confidence >= 0.70 → OPEN
- 0.50 to 0.70 → LIKELY_OPEN
- 0.30 to 0.50 → UNCERTAIN
- 0.15 to 0.30 → LIKELY_BLOCKED
- < 0.15 → BLOCKED

These values are starting points, not authoritative truth labels.

## 7. Graph construction

The graph is built from OSM road segments, not from a raster skeleton.

- each OSM road segment becomes a graph edge
- road intersections and endpoints become nodes
- edge weight is road length in meters
- OSM attributes are preserved on the edge, including osm_id and highway
- model confidence and status are stored on every edge
- CRS remains EPSG:32643

If the OSM GeoPackage does not explicitly represent intersection nodes, the pipeline converts line endpoints and line-line intersections into graph nodes by computing geometric intersections between candidate road segments.

## 8. Criticality analysis

The graph analysis includes:
- degree centrality
- betweenness centrality
- weighted betweenness using road length
- edge betweenness centrality
- bridges
- articulation points

This identifies:
- top 10 critical road segments
- top 10 critical nodes
- potential gatekeeper roads
- vulnerable corridors

## 9. Failure simulation

Three scenario classes are implemented:

1. Remove the road completely
2. Increase travel cost / reduce effective capacity by multiplying length
3. Force a road to BLOCKED if model confidence is below threshold

For each scenario, the pipeline measures:
- number of connected components
- largest connected component
- connectivity ratio
- average shortest path length, with disconnected cases handled carefully
- total increase in route distance
- number of disconnected node pairs
- percentage of network affected

Disconnected graphs are handled by restricting shortest-path statistics to the largest connected component when the graph is fragmented. In other words, average shortest path is only computed on a meaningful connected subgraph; disconnected pairs are tracked separately instead of forcing a misleading average over the whole graph.

## 10. Resilience score

The resilience metric is configured as a weighted combination of:
- connectivity retention
- shortest-path retention
- largest-component retention
- critical-route availability

This is intentionally not a single deterministic formula for all urban systems; the weights are adjustable via configuration so the score can be tuned to the application.

## 11. Important edge cases handled

The pipeline explicitly handles:
- roads outside raster extent
- nodata and NaN samples
- duplicate geometries
- invalid geometries
- zero-length geometries
- intersections and junctions
- disconnected graph components
- raster boundary sampling
- multilinestrings
- missing OSM attributes
- very short roads

## 12. Execution order

Run the modules in this order:

1. `python 01_fuse_v5_with_osm.py --osm bengaluru_large_roads_utm.gpkg --raster bengaluru_v5_probability.tif --output road_edge_confidence.gpkg`
2. `python 02_build_osm_graph.py --input road_edge_confidence.gpkg --output bengaluru_road_graph.gpickle`
3. `python 03_criticality_analysis.py --graph bengaluru_road_graph.gpickle --roads-output critical_roads.csv --nodes-output critical_nodes.csv --critical-gpkg top_critical_roads.gpkg`
4. `python 04_failure_simulation.py --graph bengaluru_road_graph.gpickle --critical-roads critical_roads.csv --output resilience_results.csv`
5. `python 05_resilience_metrics.py --input resilience_results.csv --output resilience_summary.csv`
6. `python 06_visualize_results.py --confidence road_edge_confidence.gpkg --critical-gpkg top_critical_roads.gpkg --graph bengaluru_road_graph.gpickle --output-dir visualizations`

## 13. Assumptions and limitations

- OSM road centerlines are treated as the primary network backbone.
- V5 segmentation is probabilistic evidence, not ground truth.
- The model Dice of about 0.5013 indicates the probability map is useful for evidence weighting but not absolute truth labeling.
- The road-confidence status is a heuristic label requiring validation against field or map data.
- Some OSM features may be topologically simplified or missing urban access roads.

## 14. Expected outputs

The pipeline should generate:
- road confidence per OSM segment
- a graph with edges carrying model confidence and status
- a ranked list of critical roads and nodes
- resilience metrics under simulated road failures
- visualizations for confidence, critical roads, and an example block scenario

## 15. Notes for this project

The current V5 skeleton graph is not the correct foundation for the final resilience system. The OSM backbone approach is the correct architecture for Bengaluru because it preserves the urban mobility topology and ties model confidence to each road segment rather than using a fragmented semantic skeleton as the network graph.
