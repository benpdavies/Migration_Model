import json
import numpy as np

with open('ben/oa_2011_centroids.geojson') as f:
    data = json.load(f)

geo_codes = []
coords = []

for feature in data['features']:
    geo_codes.append(feature['properties']['geo_code'])
    coords.append(feature['geometry']['coordinates'])

num_areas = len(coords)
distances = np.zeros((num_areas, num_areas))
distance_dict = {}

for i in range(2):
    for j in range(2):
        # if i != j:

        code1 = geo_codes[i]
        c1 = coords[i]
        code2 = geo_codes[j]
        c2 = coords[j]
        label = str(code1) + str(code2)

        dist = np.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)
        distances[i, j] = dist
        distance_dict[label] = dist

print(distance_dict)


