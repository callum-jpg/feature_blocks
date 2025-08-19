from feature_blocks.features._extract import _get_model
import numpy

def test_dummy_model_inference():
    model = _get_model("dummy")

    # Dummy model can take in any shape - it's not actually used
    output = model([1, 2, 3]) 

    assert output.shape == model.output_shape

def test_conv_model_inference():
    model = _get_model("dummy")
    
    input_data = numpy.random.randint(0, 255, size=(1, 1, 256, 256))

    output = model(
        input_data
    )

    assert output.shape == model.output_shape


def test_lbp_features():
    model = _get_model("lbp")
    
    input_data = numpy.random.randint(0, 255, size=(1, 1, 256, 256))

    output = model(
        input_data
    )

    assert output.shape == model.output_shape

