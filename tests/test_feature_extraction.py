import functools

import numpy
import geopandas
import skimage

def random_shapely_circles(
    image_shape, num_circles, min_radius=15, max_radius=35, seed=None, pixel_size=1
):
    """Generate a list of randomly sized shapely Polygons.

    pixel_size: This is used for testing as Xenium spatial transcriptomic
    datasets scale their segmentation polygons to the micron coordindate system.
    For Xenium, there are 0.2125 pixels per micron, so we scale to this
    """
    circles = []

    numpy.random.seed(seed=seed)

    for _ in range(num_circles):
        x_center = numpy.random.uniform(0, image_shape[1]) * pixel_size
        y_center = numpy.random.uniform(0, image_shape[0]) * pixel_size
        radius = numpy.random.uniform(min_radius, max_radius) * pixel_size

        circle = shapely.geometry.Point(x_center, y_center).buffer(radius)

        circles.append(circle)

    return circles

def get_labels_from_polygons(
    polygons,
    crop_window_size,
    pixel_size: float = 0.2125,
) -> numpy.array:
    """
    Rasterize labels from a GeoPandas DataFrame.

    If rasterising a cropped region, make sure that the coordinates
    have been translated to the new crop origin with translate_polygons_to_crop
    """
    output_array = numpy.zeros((crop_window_size, crop_window_size), dtype=numpy.int32)

    label_ids = numpy.arange(polygons.shape[0]) + 1

    for label_id, polygon in zip(label_ids, polygons):
        exterior_coords = numpy.array(polygon.exterior.coords)
        exterior_coords = numpy.multiply(exterior_coords, pixel_size**-1)
        rr, cc = skimage.draw.polygon(
            exterior_coords[:, 1], exterior_coords[:, 0], shape=output_array.shape
        )
        output_array[rr, cc] = label_id

    return output_array

def get_polygon_features(
    image,
    polygons,
    feature_extraction_fn,
    pixel_size,
    geometry_col_name: str = "geometry",
):
    """For a GeoPandas DataFrame of polygons,
    convert them to an integer labelmap and extract
    features of an image using feature_extraction_fn.
    
    Polygon features will be extracted independently, which
    will allow for overlap of objects, if they exist
    """

    results = {"cell_id": []}

    for cell_id, polygon in polygons.iterrows():

        label = polygon_to_mask(
            polygon=polygon[geometry_col_name],
            crop_window_size=image.shape[0],
            pixel_size=pixel_size
        )

        if numpy.count_nonzero(label) > 0:
            results["cell_id"].append(cell_id)

            features = feature_extraction_fn(image=image, labels=label)

            results = combine_dicts(results, features)

    return results

def polygon_to_mask(
    polygon,
    crop_window_size,
    pixel_size: float = 0.2125,
) -> numpy.array:
    """
    Rasterize a single polygon into a binary mask
    """
    output_array = numpy.zeros((crop_window_size, crop_window_size), dtype=numpy.int32)

    exterior_coords = numpy.array(polygon.exterior.coords)
    exterior_coords = numpy.multiply(exterior_coords, pixel_size**-1)
    rr, cc = skimage.draw.polygon(
        exterior_coords[:, 1], exterior_coords[:, 0], shape=output_array.shape
    )
    output_array[rr, cc] = True

    return output_array


class TestFeatureExtraction:
    def test_lbp_blocks_rgb(self):
        numpy.random.seed(42)
        img = numpy.random.random((4, 4, 3))

        rgb_lbp = functools.partial(
            histology_features.multichannel_apply_fn,
            feature_extraction_fn=histology_features.lbp_features,
            channel_axis=-1,
        )

        output = histology_features.feature_map_blocks2(
            img,
            rgb_lbp,
            window_size=[2, 2, 3],
            radius=[1],
        )

        expected_output = numpy.array(
            [
                [
                    [[5.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 4.0, 4.0, 3.0, 3.0, 0.0]],
                    [[5.0, 0.0, 1.0, 0.0, 6.0, 0.0, 0.0, 4.0, 0.0, 3.0, 3.0, 3.0]],
                ],
                [
                    [[1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 4.0, 0.0, 3.0, 3.0, 0.0]],
                    [[1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 3.0, 3.0, 0.0]],
                ],
            ]
        )

        numpy.testing.assert_equal(output, expected_output)

    def test_lbp_blocks_grayscale(self):
        numpy.random.seed(42)
        img = numpy.random.random((4, 4))

        output = histology_features.feature_map_blocks2(
            img,
            histology_features.lbp_features,
            window_size=[2, 2],
            radius=[1],
        )

        expected_output = numpy.array(
            [
                [[1.0, 0.0, 3.0, 3.0], [0.0, 0.0, 4.0, 0.0]],
                [[1.0, 0.0, 0.0, 3.0], [5.0, 0.0, 1.0, 3.0]],
            ]
        )

        numpy.testing.assert_equal(output, expected_output)

    def test_haralick_blocks(self):
        numpy.random.seed(42)
        img = numpy.random.randint(255, size=(2, 2))

        output = histology_features.feature_map_blocks2(
            img,
            histology_features.haralick_features,
            window_size=[2, 2],
        )

        expected_output = numpy.array(
            [
                [
                    [
                        3.75000000e-01,
                        8.74550000e03,
                        -7.19803846e-01,
                        2.66490625e03,
                        1.34914053e-03,
                        1.93500000e02,
                        1.91412500e03,
                        5.00000000e-01,
                        1.50000000e00,
                        4.13580247e-03,
                        5.00000000e-01,
                        -1.00000000e00,
                        9.60336677e-01,
                    ]
                ]
            ]
        )

        numpy.testing.assert_almost_equal(output, expected_output)

    def test_mean_overlap_2d(self):
        # Make a checkerboard iamge
        img = numpy.indices((6, 6)).sum(axis=0) % 2

        output = histology_features.feature_map_overlap_blocks(
            img,
            feature_extract_function=lambda x: [numpy.mean(x)],
            window_size=[3, 3],
            overlap=0.25
        )

        expected_output = numpy.array(
            [[[0.44444444],
            [0.55555556]],

            [[0.55555556],
            [0.44444444]]]
        )

        numpy.testing.assert_almost_equal(output, expected_output)

class TestPolygonFeatureExtraction:
    def extract_image_intensity(
            self,
            image,
            labels
    ):
        features = skimage.measure.regionprops_table(
            labels, 
            intensity_image=image, 
            properties=["intensity_mean", "intensity_max", "intensity_min"]
        )

        return features

    def test_polygon_indexing(self):
        """Ensure that when extracting features from polygon masks
        that the indexing matches the original polygon input.
        
        This is a codepath encountered when using Xenium segmentation masks,
        which are polygons, with traditional image feature extraction methods
        that expect integer based label arrays."""

        NUM_OBJECTS = 5
        IMAGE_SIZE = 224

        circle_polygons = random_shapely_circles(
                (IMAGE_SIZE, IMAGE_SIZE), 
                num_circles=NUM_OBJECTS, 
                min_radius=25, 
                max_radius=25, 
                seed=42
                )

        # Convert list of polygons into a GeoSeries, which is the standard
        # format for Xenium polygons.
        # Make geoDataFrame
        test_polygons = geopandas.GeoDataFrame(
            {"geometry": circle_polygons}
        )
        # Extract geometry series
        test_polygons = test_polygons["geometry"]

        # Covnert polygons into an instance labelmap
        # The polygon instance ID will be polygon_index + 1
        instance_labels = get_labels_from_polygons(
            polygons=test_polygons,
            crop_window_size=IMAGE_SIZE,
            # Here we set pixel size to 1, since we are already
            # in pixel space. If it were Xenium polygons,
            # pixel_size would be 0.2125
            pixel_size=1
        )

        # As a test, extract intensity for the label map to ensure
        # that indieces match
        # ie. we are going to record object intensity, which should 
        # just be the input label ID.
        data = get_polygon_features(
            image=instance_labels,
            polygons=test_polygons,
            feature_extraction_fn=self.extract_image_intensity,
            pixel_size=1
        )
        
        expected_output = numpy.arange(1, NUM_OBJECTS + 1)
        output = data["intensity_max"]

        numpy.testing.assert_almost_equal(output, expected_output)


