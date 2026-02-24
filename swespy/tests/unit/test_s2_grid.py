from swespy.data.s2_grid import cell_id_to_lat_lng, lat_lng_to_cell_id
from swespy.model.evaluate import haversine_distance


def test_s2_round_trip_center_distance_under_1km() -> None:
    lat, lng = 59.3293, 18.0686
    token = lat_lng_to_cell_id(lat, lng, level=14)
    center_lat, center_lng = cell_id_to_lat_lng(token)
    distance_km = haversine_distance(lat, lng, center_lat, center_lng)
    assert distance_km < 1.0
