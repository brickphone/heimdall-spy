import json

from eval import walk_routes


def test_routes_rank_street_gaps_and_ignore_covered_or_streetless_cells(tmp_path):
    south, west, dlat, dlng = 58.40, 15.60, 0.00225, 0.0043  # ~250 m cells
    counts = [[0, 0, 9], [0, 50, 0], [0, 0, 0]]
    (tmp_path / "density-grid.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "outside_mask": [[False] * 3] * 3,
                "south": south,
                "west": west,
                "dlat": dlat,
                "dlng": dlng,
                "cell_m": 250,
            }
        )
    )

    def street(j, i, name):  # ~240 m east-west street through cell (j, i).
        lat = south + (j + 0.5) * dlat
        return {
            "name": name,
            "highway": "residential",
            "geometry": [(lat, west + i * dlng + 0.0001), (lat, west + (i + 1) * dlng - 0.0001)],
        }

    ways = [
        street(0, 0, "A-gatan"),
        street(0, 1, "B-gatan"),  # Two adjacent gap cells
        street(1, 1, "Covered"),  # 50 references: not a gap
        street(2, 2, "C-vägen"),  # Isolated gap cell
        street(0, 2, "Busy"),
    ]  # 9 references: not a gap
    (tmp_path / "osm-walkable.json").write_text(json.dumps(ways))
    report = walk_routes.propose(tmp_path, 3)
    routes = report["routes"]
    assert [r["streets"] for r in routes] == [["A-gatan", "B-gatan"], ["C-vägen"]]
    assert routes[0]["gap_cells_250m"] == 2 and routes[0]["existing_references"] == 0
    assert (tmp_path / "walk-routes.html").exists()


def test_pedestrian_mode_proposes_only_unserved_footways(tmp_path):
    lat0, lng0 = 58.41, 15.62
    dlng = 0.0001
    served_way = {
        "name": "Served-gången",
        "highway": "footway",
        "geometry": [(lat0, lng0 + i * dlng) for i in range(0, 40, 4)],
    }
    lonely_way = {
        "name": "Ensam-stigen",
        "highway": "path",
        "geometry": [(lat0 + 0.01, lng0 + i * dlng) for i in range(0, 40, 4)],
    }
    road = {
        "name": "Bilvägen",
        "highway": "residential",
        "geometry": [(lat0 + 0.02, lng0 + i * dlng) for i in range(0, 40, 4)],
    }
    (tmp_path / "osm-walkable.json").write_text(json.dumps([served_way, lonely_way, road]))
    refs = [{"lat": lat0, "lng": lng0 + i * dlng} for i in range(0, 40)]  # Along the first.
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"references_in_index_order": refs}))
    report = walk_routes.propose_pedestrian(tmp_path, index, 3)
    assert [r["streets"] for r in report["routes"]] == [["Ensam-stigen"]]
    assert report["routes"][0]["unserved_pedestrian_km"] > 0.15
